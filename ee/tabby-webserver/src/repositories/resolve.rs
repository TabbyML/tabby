use std::{path::PathBuf, str::FromStr, sync::Arc};

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

use crate::{path::repository_meta_db, schema::repository::RepositoryService};

#[derive(Clone)]
pub struct RepositoryCache {
    cache: Store,
}

impl std::fmt::Debug for RepositoryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RepositoryCache").finish()
    }
}
type RepositoryBucket<'a> = Bucket<'a, String, kv::Json<RepositoryMeta>>;

static META_BUCKET: &str = "meta";
static META_BUCKET_VERSION_KEY: &str = "version";

impl RepositoryCache {
    pub fn new() -> Result<Self> {
        let config = Config::new(repository_meta_db());
        let store = Store::new(config)?;
        Ok(RepositoryCache { cache: store })
    }

    pub fn latest_version(&self) -> Result<u64> {
        let bucket: Bucket<_, String> = self.cache.bucket(Some(META_BUCKET))?;
        if !bucket.contains(&META_BUCKET_VERSION_KEY.to_string())? {
            self.update_latest_version(self.get_next_version()?)?;
        }
        Ok(bucket
            .get(&META_BUCKET_VERSION_KEY.to_string())?
            .expect("Cache version must always be set")
            .parse()?)
    }

    pub fn update_latest_version(&self, version: u64) -> Result<()> {
        let bucket = self.cache.bucket(Some(META_BUCKET))?;
        bucket.set(&META_BUCKET_VERSION_KEY.to_string(), &version.to_string())?;
        self.clear_versions_under(version)?;
        Ok(())
    }

    pub fn get_next_version(&self) -> Result<u64> {
        Ok(self.cache.generate_id()?)
    }

    fn versioned_bucket(&self, version: u64) -> Result<RepositoryBucket> {
        let bucket_name = format!("repositories_{}", version);
        Ok(self.cache.bucket(Some(&bucket_name))?)
    }

    fn bucket(&self) -> Result<RepositoryBucket> {
        self.versioned_bucket(self.latest_version()?)
    }

    pub fn clear_versions_under(&self, old_version: u64) -> Result<()> {
        for bucket in self.cache.buckets() {
            let Some((_, version)) = bucket.split_once('_') else {
                continue;
            };
            let Ok(version) = version.parse::<u64>() else {
                continue;
            };
            if version < old_version {
                self.cache.drop_bucket(bucket)?;
            }
        }
        Ok(())
    }

    pub fn add_repository_meta(&self, version: u64, file: RepositoryMeta) -> Result<()> {
        let key = format!("{}:{}", file.git_url, file.filepath);
        self.versioned_bucket(version)?.set(&key, &kv::Json(file))?;
        Ok(())
    }

    pub fn get_repository_meta(&self, git_url: &str, filepath: &str) -> Result<RepositoryMeta> {
        let key = format!("{git_url}:{filepath}");
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

pub struct RepositoryResolver {
    cache: RepositoryCache,
    service: Arc<dyn RepositoryService>,
}

impl RepositoryResolver {
    pub fn new(cache: RepositoryCache, service: Arc<dyn RepositoryService>) -> Self {
        RepositoryResolver { cache, service }
    }

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

    pub async fn resolve_meta(&self, key: &RepositoryKey) -> Result<RepositoryMeta> {
        let git_url = self
            .service
            .get_repository_by_name(key.repo_name.clone())
            .await?
            .git_url;
        self.cache
            .get_repository_meta(&git_url, &key.rel_path)
            .map(RepositoryMeta::from)
    }

    pub fn resolve_all(&self) -> Result<Response> {
        let mut entries = vec![];
        for entry in self.cache.bucket()?.iter() {
            let key: String = entry?.key()?;
            let Some(key) = RepositoryCache::str_to_key(&key) else {
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
        for entry in self.cache.bucket()?.iter() {
            let entry = entry?;
            let key: String = entry.key()?;
            let Some(key) = RepositoryCache::str_to_key(&key) else {
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
