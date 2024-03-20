use std::sync::Arc;

use axum::{
    extract::{Path, State},
    middleware::from_fn_with_state,
    response::Response,
    routing, Router,
};
use hyper::{Body, StatusCode};

use crate::{hub::require_login_middleware, schema::auth::AuthenticationService, service::AsID};

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
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;
    Ok(Response::new(Body::from(avatar.into_vec())))
}
