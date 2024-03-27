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
use tracing::warn;

use crate::schema::{
    auth::AuthenticationService,
    repository::{RepositoryMeta, RepositoryService, ResolveParams},
};

pub type ResolveState = Arc<dyn RepositoryService>;

pub fn routes(rs: ResolveState, auth: Arc<dyn AuthenticationService>) -> Router {
    Router::new()
        .route("/resolve", routing::get(resolve))
        .route("/resolve/", routing::get(resolve))
        .route("/:name/resolve/.git/", routing::get(not_found))
        .route("/:name/resolve/.git/*path", routing::get(not_found))
        .route("/:name/resolve/", routing::get(resolve_path))
        .route("/:name/resolve/*path", routing::get(resolve_path))
        .route("/:name/meta/", routing::get(meta))
        .route("/:name/meta/*path", routing::get(meta))
        .with_state(rs.clone())
        .fallback(not_found)
        .layer(from_fn_with_state(auth, require_login_middleware))
}

async fn require_login_middleware(
    State(auth): State<Arc<dyn AuthenticationService>>,
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

    let Ok(_) = auth.verify_access_token(&token).await else {
        return unauthorized;
    };

    next.run(request).await
}

async fn not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}

async fn resolve_path(
    State(rs): State<ResolveState>,
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

async fn meta(
    State(rs): State<ResolveState>,
    Path(repo): Path<ResolveParams>,
) -> Result<Json<RepositoryMeta>, StatusCode> {
    let key = repo.dataset_key();
    if let Some(resp) = rs.resolve_meta(&key).await {
        return Ok(Json(resp));
    }
    Err(StatusCode::NOT_FOUND)
}

async fn resolve(State(rs): State<ResolveState>) -> Result<Response, StatusCode> {
    rs.resolve_all().await.map_err(|_| StatusCode::NOT_FOUND)
}
