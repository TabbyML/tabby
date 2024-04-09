mod oauth;
mod resolve;

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Response,
    routing, Json, Router,
};
pub use resolve::RepositoryCache;
use tracing::{instrument, warn};

use crate::{
    repositories::resolve::{RepositoryMeta, ResolveParams},
    schema::{
        auth::AuthenticationService, github_repository_provider::GithubRepositoryProviderService,
        setting::SettingService,
    },
};

use self::oauth::OAuthState;

pub type ResolveState = Arc<RepositoryCache>;

pub fn routes(
    rs: Arc<ResolveState>,
    auth: Arc<dyn AuthenticationService>,
    settings: Arc<dyn SettingService>,
    github_repository_provider: Arc<dyn GithubRepositoryProviderService>,
) -> Router {
    let oauth_state = OAuthState {
        settings,
        github_repository_provider,
    };
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
        .nest("/oauth", oauth::routes(oauth_state))
        .fallback(not_found)
    // .layer(from_fn_with_state(auth, require_login_middleware))
}

async fn not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}

#[instrument(skip(repo))]
async fn resolve_path(
    State(rs): State<Arc<ResolveState>>,
    Path(repo): Path<ResolveParams>,
) -> Result<Response, StatusCode> {
    let Some(conf) = rs.find_repository(repo.name_str()) else {
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

#[instrument(skip(repo))]
async fn meta(
    State(rs): State<Arc<ResolveState>>,
    Path(repo): Path<ResolveParams>,
) -> Result<Json<RepositoryMeta>, StatusCode> {
    let key = repo.dataset_key();
    if let Some(resp) = rs.resolve_meta(&key) {
        return Ok(Json(resp));
    }
    Err(StatusCode::NOT_FOUND)
}

async fn resolve(State(rs): State<Arc<ResolveState>>) -> Result<Response, StatusCode> {
    rs.resolve_all().await.map_err(|_| StatusCode::NOT_FOUND)
}
