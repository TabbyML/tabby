mod resolve;

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    middleware::from_fn_with_state,
    response::Response,
    routing, Router,
};
use tracing::{instrument, warn};

use self::resolve::ResolveState;
use crate::{
    handler::require_login_middleware,
    repositories::resolve::ResolveParams,
    schema::{auth::AuthenticationService, git_repository::GitRepositoryService},
};

pub fn routes(
    repository: Arc<dyn GitRepositoryService>,
    auth: Arc<dyn AuthenticationService>,
) -> Router {
    Router::new()
        .route("/resolve", routing::get(resolve))
        .route("/resolve/", routing::get(resolve))
        .route("/:name/resolve/.git/", routing::get(not_found))
        .route("/:name/resolve/.git/*path", routing::get(not_found))
        .route("/:name/resolve/", routing::get(resolve_path))
        .route("/:name/resolve/*path", routing::get(resolve_path))
        .with_state(Arc::new(ResolveState::new(repository)))
        .fallback(not_found)
        .layer(from_fn_with_state(auth, require_login_middleware))
}

async fn not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}

#[instrument(skip(rs))]
async fn resolve_path(
    State(rs): State<Arc<ResolveState>>,
    Path(repo): Path<ResolveParams>,
) -> Result<Response, StatusCode> {
    let Some(conf) = rs.find_repository(repo.name_str()).await else {
        return Err(StatusCode::NOT_FOUND);
    };
    let root = conf.dir();
    let full_path = root.join(repo.os_path());
    let is_dir = tokio::fs::metadata(full_path.clone())
        .await
        .map(|m| m.is_dir())
        .unwrap_or(false);

    if is_dir {
        return match rs.resolve_dir(&repo, root, full_path.clone()).await {
            Ok(resp) => Ok(resp),
            Err(err) => {
                warn!("failed to resolve_dir <{:?}>: {}", full_path, err);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        };
    }

    match rs.resolve_file(root, &repo).await {
        Ok(resp) => Ok(resp),
        Err(err) => {
            warn!("failed to resolve_file <{:?}>: {}", full_path, err);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn resolve(State(rs): State<Arc<ResolveState>>) -> Result<Response, StatusCode> {
    rs.resolve_all().await.map_err(|_| StatusCode::NOT_FOUND)
}
