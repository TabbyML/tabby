use std::{collections::HashMap, path::PathBuf, str::FromStr};

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

lazy_static! {
    static ref META: HashMap<DatasetKey, Meta> = load_meta();
}

const DIRECTORY_MIME_TYPE: &str = "application/vnd.directory+json";

#[derive(Hash, PartialEq, Eq, Debug)]
pub struct DatasetKey {
    local_name: String,
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
            local_name: self.name.clone(),
            rel_path: self.path_str().to_string(),
        }
    }

    pub fn name_str(&self) -> &str {
        self.name.as_str()
    }

    pub fn path_str(&self) -> &str {
        self.path.as_deref().unwrap_or("")
    }
}

#[derive(Serialize)]
struct ListDir {
    entries: Vec<String>,
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
    let iter = match SourceFile::all() {
        Ok(all) => all,
        Err(_) => {
            return dataset;
        }
    };
    for file in iter {
        if let Some(name) = repo_conf.get(&file.git_url).map(|repo| repo.name()) {
            let key = DatasetKey {
                local_name: name,
                rel_path: file.filepath.clone(),
            };
            dataset.insert(key, file.into());
        }
    }
    dataset
}

/// Resolve a directory
pub async fn resolve_dir(root: PathBuf, full_path: PathBuf) -> Result<Response> {
    let mut read_dir = tokio::fs::read_dir(full_path).await?;
    let mut entries = vec![];

    while let Some(entry) = read_dir.next_entry().await? {
        let path = entry
            .path()
            .strip_prefix(&root)?
            .to_str()
            .unwrap()
            .to_string();
        entries.push(path);
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
