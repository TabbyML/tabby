use std::{path::PathBuf, str::FromStr};

use anyhow::{anyhow, Result};
use axum::{
    body::boxed,
    http::{header, Request, Uri},
    response::{IntoResponse, Response},
    Json,
};
use hyper::Body;
use kv::{Bucket, Config, Store};
use serde::{Deserialize, Serialize};
use tabby_common::{config::RepositoryConfig, SourceFile, Tag};
use tower::ServiceExt;
use tower_http::services::ServeDir;

pub struct RepositoryCache {
    cache: Store,
}

impl std::fmt::Debug for RepositoryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RepositoryCache").finish()
    }
}
type RepositoryBucket<'a> = Bucket<'a, String, kv::Json<SourceFile>>;

impl RepositoryCache {
    pub fn new() -> Result<Self> {
        let config = Config::new(tabby_common::path::repository_meta_db());
        let store = Store::new(config)?;
        Ok(RepositoryCache { cache: store })
    }

    fn bucket(&self) -> Result<RepositoryBucket> {
        Ok(self.cache.bucket(Some("repositories"))?)
    }

    pub fn clear(&self) -> Result<()> {
        self.bucket()?.clear()?;
        Ok(())
    }

    pub fn add_repository_meta(&self, file: SourceFile) -> Result<()> {
        let key = format!("{}:{}", file.repository_name, file.filepath);
        self.bucket()?.set(&key, &kv::Json(file))?;
        Ok(())
    }

    pub fn get_repository_meta(&self, repository_name: &str, filepath: &str) -> Result<SourceFile> {
        let key = format!("{repository_name}:{filepath}");
        let Some(kv::Json(val)) = self.bucket()?.get(&key)? else {
            return Err(anyhow!("Repository meta not found"));
        };
        Ok(val)
    }

    fn str_to_key(str: &str) -> Option<RepositoryKey> {
        str.split_once(':').map(|(name, path)| RepositoryKey {
            repo_name: name.to_string(),
            rel_path: path.to_string(),
        })
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

/// Webserver resolve functions
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

    pub fn resolve_meta(&self, key: &RepositoryKey) -> Result<RepositoryMeta> {
        self.get_repository_meta(&key.repo_name, &key.rel_path)
            .map(RepositoryMeta::from)
    }

    pub fn resolve_all(&self) -> Result<Response> {
        let mut entries = vec![];
        for entry in self.bucket()?.iter() {
            let key: String = entry?.key()?;
            let Some(key) = Self::str_to_key(&key) else {
                continue;
            };
            entries.push(DirEntry {
                kind: DirEntryKind::Dir,
                basename: key.repo_name,
            })
        }

        let body = Json(ListDir { entries }).into_response();
        let resp = Response::builder()
            .header(header::CONTENT_TYPE, DIRECTORY_MIME_TYPE)
            .body(body.into_body())?;

        Ok(resp)
    }

    pub fn find_repository(&self, name: &str) -> Result<RepositoryConfig> {
        for entry in self.bucket()?.iter() {
            let entry = entry?;
            let key: String = entry.key()?;
            let Some(key) = Self::str_to_key(&key) else {
                continue;
            };
            if &key.repo_name == name {
                let kv::Json(value) = entry.value()?;
                return Ok(RepositoryConfig::new_named(key.repo_name, value.git_url));
            }
        }
        Err(anyhow!("Repository not found"))
    }
}
