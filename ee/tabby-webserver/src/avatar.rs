use std::sync::Arc;

use axum::{
    extract::{Path, State},
    middleware::from_fn_with_state,
    response::Response,
    routing, Router,
};
use hyper::{header::CONTENT_TYPE, Body, StatusCode};
use tracing::error;

use crate::{
    handler::require_login_middleware, schema::auth::AuthenticationService, service::AsID,
};

pub fn routes(auth: Arc<dyn AuthenticationService>) -> Router {
    Router::new()
        .route("/:id", routing::get(avatar))
        .with_state(auth.clone())
        .layer(from_fn_with_state(auth, require_login_middleware))
}

pub async fn avatar(
    State(state): State<Arc<dyn AuthenticationService>>,
    Path(id): Path<i64>,
) -> Result<Response<Body>, StatusCode> {
    let avatar = state
        .get_user_avatar(&id.as_id())
        .await
        .map_err(|e| {
            error!("Failed to retrieve avatar: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .ok_or(StatusCode::NOT_FOUND)?;
    let mut response = Response::new(Body::from(avatar.into_vec()));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, "image/jpeg".parse().unwrap());
    Ok(response)
}
