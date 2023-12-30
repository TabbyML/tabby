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
use lazy_static::lazy_static;
use juniper_axum::extract::AuthBearer;
use tracing::{instrument, warn};
use tabby_common::config::Config;

use crate::{
    repositories,
    repositories::resolve::{resolve_dir, resolve_file, resolve_meta, Meta, ResolveParams},
    schema::ServiceLocator,
};

lazy_static! {
    static ref CONF: Config = Config::load().unwrap_or_default();
}

pub fn routes(locator: Arc<dyn ServiceLocator>) -> Router {
    Router::new()
        .route("/:name/resolve/.git/", routing::get(not_found))
        .route("/:name/resolve/.git/*path", routing::get(not_found))
        .route("/:name/resolve/", routing::get(repositories::resolve))
        .route("/:name/resolve/*path", routing::get(repositories::resolve))
        .route("/:name/meta/", routing::get(repositories::meta))
        .route("/:name/meta/*path", routing::get(repositories::meta))
        .fallback(not_found)
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

async fn not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}

#[instrument(skip(repo))]
async fn resolve(Path(repo): Path<ResolveParams>) -> Result<Response, StatusCode> {
    let Some(conf) = CONF.find_repository(repo.name_str()) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let root = conf.dir();
    let full_path = root.join(repo.os_path());
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
