mod resolve;

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    middleware::from_fn_with_state,
    response::Response,
    routing, Extension, Router,
};
use resolve::{ResolveParams, ResolveState};
use tabby_schema::{
    auth::AuthenticationService, policy::AccessPolicy, repository::RepositoryService,
};
use tracing::instrument;

use super::require_login_middleware;

pub fn routes(
    repository: Arc<dyn RepositoryService>,
    auth: Arc<dyn AuthenticationService>,
) -> Router {
    Router::new()
        .route("/:kind/:id/resolve/", routing::get(resolve_path))
        .route("/:kind/:id/resolve/*path", routing::get(resolve_path))
        // Routes support viewing a specific revision of a repository
        .route("/:kind/:id/rev/:rev/", routing::get(resolve_path))
        .route("/:kind/:id/rev/:rev/*path", routing::get(resolve_path))
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
    Path(params): Path<ResolveParams>,
    Extension(access_policy): Extension<AccessPolicy>,
) -> Result<Response, StatusCode> {
    rs.resolve(&access_policy, params).await
}
