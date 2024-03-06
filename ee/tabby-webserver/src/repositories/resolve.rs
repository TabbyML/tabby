use std::{
    collections::HashMap,
    ops::Deref,
    path::PathBuf,
    str::FromStr,
    sync::{Arc, RwLock},
};

use anyhow::Result;
use axum::{
    body::boxed,
    http::{header, Request, Uri},
    response::{IntoResponse, Response},
    Json,
};
use hyper::Body;
use serde::{Deserialize, Serialize};
use tabby_common::{config::RepositoryConfig, SourceFile, Tag};
use tower::ServiceExt;
use tower_http::services::ServeDir;
use tracing::{debug, error, warn};

use crate::{
    cron::{CronEvents, StartListener},
    schema::repository::RepositoryService,
};

pub struct RepositoryCache {
    repository_lookup: RwLock<HashMap<RepositoryKey, RepositoryMeta>>,
    service: Arc<dyn RepositoryService>,
}

impl std::fmt::Debug for RepositoryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RepositoryCache")
            .field("repository_lookup", &self.repository_lookup)
            .finish()
    }
}

impl RepositoryCache {
    pub async fn new_initialized(
        service: Arc<dyn RepositoryService>,
        events: &CronEvents,
    ) -> Arc<RepositoryCache> {
        let cache = RepositoryCache {
            repository_lookup: Default::default(),
            service,
        };
        if let Err(e) = cache.reload().await {
            error!("Failed to load repositories: {e}");
        };
        let cache = Arc::new(cache);
        cache.start_reload_listener(events);
        cache
    }

    pub async fn reload(&self) -> Result<()> {
        let new_repositories = self
            .service
            .list_repositories(None, None, None, None)
            .await?
            .into_iter()
            .map(|repository| RepositoryConfig::new_named(repository.name, repository.git_url))
            .collect();
        let mut repository_lookup = self.repository_lookup.write().unwrap();
        debug!("Reloading repositoriy metadata...");
        *repository_lookup = load_meta(new_repositories);
        Ok(())
    }

    fn start_reload_listener(self: &Arc<Self>, events: &CronEvents) {
        let clone = self.clone();
        events.scheduler_job_succeeded.start_listener(move |_| {
            let clone = clone.clone();
            async move {
                if let Err(e) = clone.reload().await {
                    warn!("Error when reloading repository cache: {e}");
                };
            }
        });
    }

    fn repositories(&self) -> impl Deref<Target = HashMap<RepositoryKey, RepositoryMeta>> + '_ {
        self.repository_lookup.read().unwrap()
    }
}

const DIRECTORY_MIME_TYPE: &str = "application/vnd.directory+json";

#[derive(Hash, PartialEq, Eq, Debug)]
pub struct RepositoryKey {
    repo_name: String,
    rel_path: String,
}

#[derive(Deserialize, Debug)]
pub struct ResolveParams {
    name: String,
    path: Option<String>,
}

impl ResolveParams {
    pub fn dataset_key(&self) -> RepositoryKey {
        RepositoryKey {
            repo_name: self.name.clone(),
            rel_path: self.os_path(),
        }
    }

    pub fn name_str(&self) -> &str {
        self.name.as_str()
    }

    pub fn path_str(&self) -> &str {
        self.path.as_deref().unwrap_or("")
    }

    pub fn os_path(&self) -> String {
        if cfg!(target_os = "windows") {
            self.path.clone().unwrap_or_default().replace('/', r"\")
        } else {
            self.path.clone().unwrap_or_default()
        }
    }
}

#[derive(Serialize)]
struct ListDir {
    entries: Vec<DirEntry>,
}

#[derive(Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum DirEntryKind {
    File,
    Dir,
}

#[derive(Serialize)]
struct DirEntry {
    kind: DirEntryKind,
    basename: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RepositoryMeta {
    git_url: String,
    filepath: String,
    language: String,
    max_line_length: usize,
    avg_line_length: f32,
    alphanum_fraction: f32,
    tags: Vec<Tag>,
}

impl From<SourceFile> for RepositoryMeta {
    fn from(file: SourceFile) -> Self {
        Self {
            git_url: file.git_url,
            filepath: file.filepath,
            language: file.language,
            max_line_length: file.max_line_length,
            avg_line_length: file.avg_line_length,
            alphanum_fraction: file.alphanum_fraction,
            tags: file.tags,
        }
    }
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

impl RepositoryCache {
    /// Resolve a directory
    pub async fn resolve_dir(
        &self,
        repo: &ResolveParams,
        root: PathBuf,
        full_path: PathBuf,
    ) -> Result<Response> {
        let mut read_dir = tokio::fs::read_dir(full_path).await?;
        let mut entries: Vec<DirEntry> = vec![];

        while let Some(entry) = read_dir.next_entry().await? {
            let basename = entry
                .path()
                .strip_prefix(&root)?
                .to_str()
                .unwrap()
                .to_string();

            let meta = entry.metadata().await?;

            let kind = if meta.is_dir() {
                DirEntryKind::Dir
            } else if meta.is_file() {
                let key = RepositoryKey {
                    repo_name: repo.name_str().to_string(),
                    rel_path: basename.clone(),
                };
                if !self.contains_meta(&key) {
                    continue;
                }
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
            .body(body.into_body())?;

        Ok(resp)
    }

    /// Resolve a file
    pub async fn resolve_file(&self, root: PathBuf, repo: &ResolveParams) -> Result<Response> {
        let uri = if !repo.path_str().starts_with('/') {
            let path = format!("/{}", repo.path_str());
            Uri::from_str(path.as_str())?
        } else {
            Uri::from_str(repo.path_str())?
        };

        let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
        let resp = ServeDir::new(root).oneshot(req).await?;

        Ok(resp.map(boxed))
    }

    pub fn resolve_meta(&self, key: &RepositoryKey) -> Option<RepositoryMeta> {
        if let Some(meta) = self.repositories().get(key) {
            return Some(meta.clone());
        }
        None
    }

    pub fn contains_meta(&self, key: &RepositoryKey) -> bool {
        self.repositories().contains_key(key)
    }

    pub fn resolve_all(&self) -> Result<Response> {
        let entries: Vec<_> = self
            .repository_lookup
            .read()
            .unwrap()
            .keys()
            .map(|repo| DirEntry {
                kind: DirEntryKind::Dir,
                basename: repo.repo_name.clone(),
            })
            .collect();

        let body = Json(ListDir { entries }).into_response();
        let resp = Response::builder()
            .header(header::CONTENT_TYPE, DIRECTORY_MIME_TYPE)
            .body(body.into_body())?;

        Ok(resp)
    }

    pub fn find_repository(&self, name: &str) -> Option<RepositoryConfig> {
        let repository_lookup = self.repository_lookup.read().unwrap();
        let key = repository_lookup
            .keys()
            .find(|repo| repo.repo_name == name)?;
        let value = repository_lookup.get(key)?;
        Some(RepositoryConfig::new_named(
            key.repo_name.clone(),
            value.git_url.clone(),
        ))
    }
}
