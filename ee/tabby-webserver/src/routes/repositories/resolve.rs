use std::{path::PathBuf, str::FromStr, sync::Arc};

use anyhow::Result;
use axum::{
    body::Body,
    http::{header, Request, Uri},
    response::{IntoResponse, Response},
    Json,
};
use juniper::ID;
use serde::{Deserialize, Serialize};
use tabby_schema::repository::{RepositoryKind, RepositoryService};
use tower::ServiceExt;
use tower_http::services::ServeDir;
use url::Url;

const DIRECTORY_MIME_TYPE: &str = "application/vnd.directory+json";

#[derive(Deserialize, Debug)]
pub struct ResolveParams {
    pub kind: RepositoryKind,
    pub id: ID,
    path: Option<String>,
}

impl ResolveParams {
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

pub(super) struct ResolveState {
    service: Arc<dyn RepositoryService>,
}

impl ResolveState {
    pub fn new(service: Arc<dyn RepositoryService>) -> Self {
        Self { service }
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
        let path = if !repo.path_str().starts_with('/') {
            format!("/{}", repo.path_str())
        } else {
            repo.path_str().to_string()
        };

        let encoded_path = urlencoding::encode(&path);

        let uri = Uri::from_str(&encoded_path)?;

        let req = Request::builder().uri(uri).body(Body::empty())?;
        let resp = ServeDir::new(root).oneshot(req).await?;

        Ok(resp.into_response())
    }

    pub async fn find_repository(&self, params: &ResolveParams) -> Option<PathBuf> {
        let repository = self
            .service
            .resolve_repository(&params.kind, &params.id)
            .await
            .ok()?;
        Some(repository.dir)
    }
}
