use std::{collections::HashMap, path::PathBuf, str::FromStr};

use anyhow::Result;
use axum::{
    body::boxed,
    http::{header, Request, Uri},
    response::{IntoResponse, Response},
    Json,
};
use hyper::Body;
use serde::{Deserialize, Serialize};
use tabby_common::{to_filename, SourceFile};
use tokio::sync::OnceCell;
use tower::ServiceExt;
use tower_http::services::ServeDir;
use tracing::error;

pub(crate) static DATASET: OnceCell<HashMap<DatasetKey, SourceFile>> = OnceCell::const_new();

const MIME_VENDOR: &str = "application/vnd.directory+json";

#[derive(Hash, PartialEq, Eq, Debug)]
pub struct DatasetKey {
    local_name: String,
    rel_path: String,
}

#[derive(Deserialize, Debug)]
pub struct Repository {
    name: String,
    path: Option<String>,
}

impl Repository {
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

/// Load dataset
pub async fn load_dataset() -> Result<()> {
    // `SourceFile::all()` depends on `std::io`, so it's blocking.
    // We need to spawn a blocking task dedicated for such scenario.
    let dataset = tokio::task::spawn_blocking(|| {
        let mut dataset = HashMap::new();
        let iter = match SourceFile::all() {
            Ok(all) => all,
            Err(err) => {
                error!("load dataset: {}", err);
                return dataset;
            }
        };
        for mut file in iter {
            let key = DatasetKey {
                local_name: to_filename(file.git_url.as_str()),
                rel_path: file.filepath.clone(),
            };
            // exclude content from response data
            file.content = "".to_string();
            dataset.insert(key, file);
        }
        dataset
    })
    .await?;

    DATASET.set(dataset)?;
    Ok(())
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
        .header(header::CONTENT_TYPE, MIME_VENDOR)
        .body(body.into_body())?;

    Ok(resp)
}

/// Resolve a file
pub async fn resolve_file(root: PathBuf, repo: &Repository) -> Result<Response> {
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
