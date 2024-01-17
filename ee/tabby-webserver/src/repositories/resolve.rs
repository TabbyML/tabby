use std::{collections::HashMap, path::PathBuf, str::FromStr, sync::Arc};

use anyhow::Result;
use axum::{
    body::boxed,
    http::{header, Request, Uri},
    response::{IntoResponse, Response},
    Json,
};
use hyper::Body;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use tabby_common::{config::Config, SourceFile, Tag};
use tower::ServiceExt;
use tower_http::services::ServeDir;

use crate::repositories::ResolveState;

lazy_static! {
    static ref META: HashMap<DatasetKey, Meta> = load_meta();
}

const DIRECTORY_MIME_TYPE: &str = "application/vnd.directory+json";

#[derive(Hash, PartialEq, Eq, Debug)]
pub struct DatasetKey {
    repo_name: String,
    rel_path: String,
}

#[derive(Deserialize, Debug)]
pub struct ResolveParams {
    name: String,
    path: Option<String>,
}

impl ResolveParams {
    pub fn dataset_key(&self) -> DatasetKey {
        DatasetKey {
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
pub struct Meta {
    git_url: String,
    filepath: String,
    language: String,
    max_line_length: usize,
    avg_line_length: f32,
    alphanum_fraction: f32,
    tags: Vec<Tag>,
}

impl From<SourceFile> for Meta {
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

/// TODO: implement auto reloading logic in future (so changes produced by tabby-scheduler command will be loaded)
fn load_meta() -> HashMap<DatasetKey, Meta> {
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
            let key = DatasetKey {
                repo_name,
                rel_path: file.filepath.clone(),
            };
            dataset.insert(key, file.into());
        }
    }
    dataset
}

/// Resolve a directory
pub async fn resolve_dir(repo_name: &str, root: PathBuf, full_path: PathBuf) -> Result<Response> {
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
            let key = DatasetKey {
                repo_name: repo_name.to_string(),
                rel_path: basename.clone(),
            };
            if !contains_meta(&key) {
                continue;
            }
            DirEntryKind::File
        } else {
            // Skip others.
            continue;
        };
        // filter out .git directory
        if kind == DirEntryKind::Dir && basename == ".git" {
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
pub async fn resolve_file(root: PathBuf, repo: &ResolveParams) -> Result<Response> {
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

pub fn resolve_meta(key: &DatasetKey) -> Option<Meta> {
    if let Some(meta) = META.get(key) {
        return Some(meta.clone());
    }
    None
}

pub fn contains_meta(key: &DatasetKey) -> bool {
    META.contains_key(key)
}

pub fn resolve_all(rs: Arc<ResolveState>) -> Result<Response> {
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
