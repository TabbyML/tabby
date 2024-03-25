use std::sync::Arc;

use axum::{
    extract::{Path, State},
    response::Response,
};
use hyper::{header::CONTENT_TYPE, Body, StatusCode};
use juniper::ID;
use tracing::error;

use crate::schema::auth::AuthenticationService;

pub async fn avatar(
    State(state): State<Arc<dyn AuthenticationService>>,
    Path(id): Path<ID>,
) -> Result<Response<Body>, StatusCode> {
    let avatar = state
        .get_user_avatar(&id)
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
