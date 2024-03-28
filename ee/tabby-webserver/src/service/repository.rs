use std::{collections::HashMap, path::PathBuf, str::FromStr, sync::Arc};

use anyhow::anyhow;
use async_trait::async_trait;
use axum::{
    body::boxed,
    http::Uri,
    response::{IntoResponse, Response},
    Json,
};
use hyper::{header, Body, Request};
use juniper::ID;
use tabby_common::{config::RepositoryConfig, Anyhow, SourceFile};
use tabby_db::DbConn;
use tokio::sync::RwLock;
use tower::ServiceExt;
use tower_http::services::ServeDir;
use tracing::{debug, error, warn};

use super::{graphql_pagination_to_filter, AsID, AsRowid};
use crate::{
    cron::{CronEvents, StartListener},
    schema::{
        repository::{
            DirEntry, DirEntryKind, FileEntry, ListDir, Repository, RepositoryKey, RepositoryMeta,
            RepositoryService, ResolveParams, DIRECTORY_MIME_TYPE,
        },
        Result,
    },
};

fn glob_matches(glob: &str, mut input: &str) -> bool {
    for part in glob.split(' ') {
        let Some((_, rest)) = input.split_once(part) else {
            return false;
        };
        input = rest;
    }
    true
}

struct RepositoryServiceImpl {
    db: DbConn,
    cache: RwLock<HashMap<RepositoryKey, RepositoryMeta>>,
}

fn load_meta(repositories: Vec<RepositoryConfig>) -> HashMap<RepositoryKey, RepositoryMeta> {
    let mut dataset = HashMap::new();
    // Construct map of String -> &RepositoryConfig for lookup
    let repo_conf = repositories
        .iter()
        .map(|repo| (repo.git_url.clone(), repo))
        .collect::<HashMap<_, _>>();
    let Ok(iter) = SourceFile::all() else {
        return dataset;
    };
    // Source files contain all metadata, read repository metadata from json
    // (SourceFile can be converted into RepositoryMeta)
    for file in iter {
        if let Some(repo_name) = repo_conf.get(&file.git_url).map(|repo| repo.name()) {
            let key = RepositoryKey {
                repo_name,
                rel_path: file.filepath.clone(),
            };
            dataset.insert(key, file.into());
        }
    }
    dataset
}

pub async fn new_repository_service(db: DbConn, initialize: bool) -> impl RepositoryService {
    let service = RepositoryServiceImpl {
        db,
        cache: Default::default(),
    };
    if initialize {
        if let Err(e) = service.refresh_cache().await {
            error!("Failed to initialize repository cache: {e}");
        }
    }
    service
}

pub fn start_reload_listener(service: Arc<dyn RepositoryService>, events: &CronEvents) {
    events.scheduler_job_succeeded.start_listener(move |_| {
        let clone = service.clone();
        async move {
            if let Err(e) = clone.refresh_cache().await {
                warn!("Error when reloading repository cache: {e}");
            };
        }
    });
}

#[async_trait]
impl RepositoryService for RepositoryServiceImpl {
    async fn list_repositories(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Repository>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let repositories = self
            .db
            .list_repositories_with_filter(limit, skip_id, backwards)
            .await?;
        Ok(repositories.into_iter().map(Into::into).collect())
    }

    async fn create_repository(&self, name: String, git_url: String) -> Result<ID> {
        Ok(self.db.create_repository(name, git_url).await?.as_id())
    }

    async fn delete_repository(&self, id: &ID) -> Result<bool> {
        Ok(self.db.delete_repository(id.as_rowid()?).await?)
    }

    async fn update_repository(&self, id: &ID, name: String, git_url: String) -> Result<bool> {
        self.db
            .update_repository(id.as_rowid()?, name, git_url)
            .await?;
        Ok(true)
    }

    async fn search_files(
        &self,
        name: String,
        path_glob: String,
        top_n: usize,
    ) -> Result<Vec<FileEntry>> {
        let git_url = self.db.get_repository_git_url(name).await?;
        let matching = self
            .cache
            .read()
            .await
            .values()
            .filter_map(|file| {
                if file.git_url == git_url && glob_matches(&path_glob, &file.filepath) {
                    Some(FileEntry {
                        r#type: "file".into(), // Directories are not currently stored in files.jsonl
                        path: file.filepath.clone(),
                    })
                } else {
                    None
                }
            })
            .take(top_n)
            .collect();
        Ok(matching)
    }

    async fn repository_meta(&self, name: String, path: String) -> Result<RepositoryMeta> {
        let git_url = self.db.get_repository_git_url(name).await?;
        self.cache
            .read()
            .await
            .values()
            .find(|&file| (file.filepath == path && file.git_url == git_url))
            .cloned()
            .ok_or_else(|| anyhow!("File not found").into())
    }

    /// Resolve a directory
    async fn resolve_dir(
        &self,
        repo: &ResolveParams,
        root: PathBuf,
        full_path: PathBuf,
    ) -> Result<Response> {
        let mut read_dir = tokio::fs::read_dir(full_path).await.anyhow()?;
        let mut entries: Vec<DirEntry> = vec![];

        while let Some(entry) = read_dir.next_entry().await.anyhow()? {
            let basename = entry
                .path()
                .strip_prefix(&root)
                .anyhow()?
                .to_str()
                .unwrap()
                .to_string();

            let meta = entry.metadata().await.map_err(anyhow::Error::from)?;

            let kind = if meta.is_dir() {
                DirEntryKind::Dir
            } else if meta.is_file() {
                let _key = RepositoryKey {
                    repo_name: repo.name_str().to_string(),
                    rel_path: basename.clone(),
                };
                DirEntryKind::File
            } else {
                // Skip others.
                continue;
            };

            // filter out .git directory at root
            if kind == DirEntryKind::Dir && basename == ".git" && repo.path.is_none() {
                continue;
            }

            entries.push(DirEntry { kind, basename });
        }

        let body = Json(ListDir { entries }).into_response();
        let resp = Response::builder()
            .header(header::CONTENT_TYPE, DIRECTORY_MIME_TYPE)
            .body(body.into_body())
            .anyhow()?;

        Ok(resp)
    }

    async fn resolve_file(&self, root: PathBuf, repo: &ResolveParams) -> Result<Response> {
        let uri = if !repo.path_str().starts_with('/') {
            let path = format!("/{}", repo.path_str());
            Uri::from_str(path.as_str()).anyhow()?
        } else {
            Uri::from_str(repo.path_str()).anyhow()?
        };

        let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
        let resp = ServeDir::new(root).oneshot(req).await.anyhow()?;

        Ok(resp.map(boxed))
    }

    async fn resolve_meta(&self, key: &RepositoryKey) -> Option<RepositoryMeta> {
        if let Some(meta) = self.cache.read().await.get(key) {
            return Some(meta.clone());
        }
        None
    }

    async fn resolve_all(&self) -> Result<Response> {
        let entries: Vec<_> = self
            .cache
            .read()
            .await
            .keys()
            .map(|repo| DirEntry {
                kind: DirEntryKind::Dir,
                basename: repo.repo_name.clone(),
            })
            .collect();

        let body = Json(ListDir { entries }).into_response();
        let resp = Response::builder()
            .header(header::CONTENT_TYPE, DIRECTORY_MIME_TYPE)
            .body(body.into_body())
            .anyhow()?;

        Ok(resp)
    }

    async fn find_repository(&self, name: &str) -> Option<RepositoryConfig> {
        let repository_lookup = self.cache.read().await;
        let key = repository_lookup
            .keys()
            .find(|repo| repo.repo_name == name)?;
        let value = repository_lookup.get(key)?;
        Some(RepositoryConfig::new_named(
            key.repo_name.clone(),
            value.git_url.clone(),
        ))
    }

    async fn refresh_cache(&self) -> Result<()> {
        let new_repositories = self
            .list_repositories(None, None, None, None)
            .await?
            .into_iter()
            .map(|repository| RepositoryConfig::new_named(repository.name, repository.git_url))
            .collect();
        let mut repository_lookup = self.cache.write().await;
        debug!("Reloading repositoriy metadata...");
        *repository_lookup = load_meta(new_repositories);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;

    use super::*;

    #[tokio::test]
    pub async fn test_duplicate_repository_error() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = new_repository_service(db, false).await;

        service
            .create_repository(
                "example".into(),
                "https://github.com/example/example".into(),
            )
            .await
            .unwrap();

        let err = service
            .create_repository(
                "example".into(),
                "https://github.com/example/example".into(),
            )
            .await
            .unwrap_err();

        assert_eq!(
            err.to_string(),
            "A repository with the same name or URL already exists"
        );
    }

    #[tokio::test]
    pub async fn test_repository_mutations() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = new_repository_service(db, false).await;

        let id_1 = service
            .create_repository(
                "example".into(),
                "https://github.com/example/example".into(),
            )
            .await
            .unwrap();

        let id_2 = service
            .create_repository(
                "example2".into(),
                "https://github.com/example/example2".into(),
            )
            .await
            .unwrap();

        service
            .create_repository(
                "example3".into(),
                "https://github.com/example/example3".into(),
            )
            .await
            .unwrap();

        assert_eq!(
            service
                .list_repositories(None, None, None, None)
                .await
                .unwrap()
                .len(),
            3
        );

        service.delete_repository(&id_1).await.unwrap();

        assert_eq!(
            service
                .list_repositories(None, None, None, None)
                .await
                .unwrap()
                .len(),
            2
        );

        service
            .update_repository(
                &id_2,
                "Example2".to_string(),
                "https://github.com/example/Example2".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(
            service
                .list_repositories(None, None, None, None)
                .await
                .unwrap()
                .first()
                .unwrap()
                .name,
            "Example2"
        );
    }
}
