pub(crate) mod repo;

use anyhow::Result;
use axum::{extract::Path, http::StatusCode, response::Response, Json};
use tabby_common::{path::repositories_dir, SourceFile};
use tracing::{debug, instrument, warn};

use crate::repositories::repo::{resolve_dir, resolve_file, Repository, DATASET};

#[instrument(skip(repo))]
pub async fn resolve(Path(repo): Path<Repository>) -> Result<Response, StatusCode> {
    debug!("repo: {:?}", repo);
    let root = repositories_dir().join(repo.name_str());
    let full_path = root.join(repo.path_str());
    let is_dir = tokio::fs::metadata(full_path.clone())
        .await
        .map(|m| m.is_dir())
        .unwrap_or(false);

    if is_dir {
        return match resolve_dir(root, full_path).await {
            Ok(resp) => Ok(resp),
            Err(err) => {
                warn!("{}", err);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        };
    }

    match resolve_file(root, &repo).await {
        Ok(resp) => Ok(resp),
        Err(err) => {
            warn!("{}", err);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

#[instrument(skip(repo))]
pub async fn meta(Path(repo): Path<Repository>) -> Result<Json<SourceFile>, StatusCode> {
    debug!("repo: {:?}", repo);
    let key = repo.dataset_key();
    if let Some(dataset) = DATASET.get() {
        if let Some(file) = dataset.get(&key) {
            return Ok(Json(file.clone()));
        }
    }
    Err(StatusCode::NOT_FOUND)
}
