mod resolve;

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Path, State},
    http::{Request, StatusCode},
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Response},
    routing, Json, Router,
};
use hyper::Body;
use juniper_axum::extract::AuthBearer;
use tabby_common::path::repositories_dir;
use tracing::{instrument, warn};

use crate::{
    repositories,
    repositories::resolve::{resolve_dir, resolve_file, resolve_meta, Meta, ResolveParams},
    schema::ServiceLocator,
};

pub fn routes(locator: Arc<dyn ServiceLocator>) -> Router {
    Router::new()
        .route("/:name/resolve/", routing::get(repositories::resolve))
        .route("/:name/resolve/*path", routing::get(repositories::resolve))
        .route("/:name/meta/", routing::get(repositories::meta))
        .route("/:name/meta/*path", routing::get(repositories::meta))
        .layer(from_fn_with_state(locator, require_login_middleware))
}

async fn require_login_middleware(
    State(locator): State<Arc<dyn ServiceLocator>>,
    AuthBearer(token): AuthBearer,
    request: Request<Body>,
    next: Next<Body>,
) -> axum::response::Response {
    let unauthorized = axum::response::Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .body(Body::empty())
        .unwrap()
        .into_response();

    let Some(token) = token else {
        return unauthorized;
    };

    let Ok(_) = locator.auth().verify_access_token(&token).await else {
        return unauthorized;
    };

    next.run(request).await
}

#[instrument(skip(repo))]
async fn resolve(Path(repo): Path<ResolveParams>) -> Result<Response, StatusCode> {
    let root = repositories_dir().join(repo.name_str());
    let full_path = root.join(repo.path_str());
    let is_dir = tokio::fs::metadata(full_path.clone())
        .await
        .map(|m| m.is_dir())
        .unwrap_or(false);

    if is_dir {
        return match resolve_dir(root, full_path.clone()).await {
            Ok(resp) => Ok(resp),
            Err(err) => {
                warn!("failed to resolve_dir <{:?}>: {}", full_path, err);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        };
    }

    match resolve_file(root, &repo).await {
        Ok(resp) => Ok(resp),
        Err(err) => {
            warn!("failed to resolve_file <{:?}>: {}", full_path, err);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

#[instrument(skip(repo))]
async fn meta(Path(repo): Path<ResolveParams>) -> Result<Json<Meta>, StatusCode> {
    let key = repo.dataset_key();
    if let Some(resp) = resolve_meta(&key) {
        return Ok(Json(resp));
    }
    Err(StatusCode::NOT_FOUND)
}
