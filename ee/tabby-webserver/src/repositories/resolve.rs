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
use tabby_common::{config::Config, SourceFile, Tag};
use tokio_cron_scheduler::{Job, JobScheduler};
use tower::ServiceExt;
use tower_http::services::ServeDir;

use crate::repositories::ResolveState;

#[derive(Debug)]
pub struct RepositoryCache {
    repositories: RwLock<HashMap<RepositoryKey, RepositoryMeta>>,
}

impl RepositoryCache {
    pub fn new_initialized() -> RepositoryCache {
        let cache = RepositoryCache {
            repositories: Default::default(),
        };
        cache.reload();
        cache
    }

    fn reload(&self) {
        let mut repositories = self.repositories.write().unwrap();
        *repositories = load_meta();
    }

    pub fn repositories(&self) -> impl Deref<Target = HashMap<RepositoryKey, RepositoryMeta>> + '_ {
        self.repositories.read().unwrap()
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

fn load_meta() -> HashMap<RepositoryKey, RepositoryMeta> {
    let mut dataset = HashMap::new();
    let repo_conf = match Config::load() {
        Ok(config) => config
            .repositories
            .into_iter()
            .map(|repo| (repo.git_url.clone(), repo))
            .collect::<HashMap<_, _>>(),
        Err(_) => {
            return dataset;
        }
    };
    let Ok(iter) = SourceFile::all() else {
        return dataset;
    };
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

    pub fn resolve_all(&self, rs: Arc<ResolveState>) -> Result<Response> {
        let entries: Vec<_> = rs
            .repositories
            .iter()
            .map(|repo| DirEntry {
                kind: DirEntryKind::Dir,
                basename: repo.name(),
            })
            .collect();

        let body = Json(ListDir { entries }).into_response();
        let resp = Response::builder()
            .header(header::CONTENT_TYPE, DIRECTORY_MIME_TYPE)
            .body(body.into_body())?;

        Ok(resp)
    }
    pub async fn start_reload_job(self: &Arc<Self>) {
        let cache = self.clone();
        let scheduler = JobScheduler::new().await.unwrap();
        scheduler
            .add(
                Job::new("0 1/5 * * * * *", move |_, _| {
                    cache.reload();
                })
                .unwrap(),
            )
            .await
            .unwrap();
        scheduler.start().await.unwrap();
    }
}
