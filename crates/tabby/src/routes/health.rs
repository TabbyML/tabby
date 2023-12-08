use std::sync::Arc;

use axum::{extract::State, Json};

use crate::services::health;

#[utoipa::path(
    get,
    path = "/v1/health",
    tag = "v1",
    responses(
        (status = 200, description = "Success", body = HealthState, content_type = "application/json"),
    ),
    security(
        ("token" = [])
    )
)]
pub async fn health(State(state): State<Arc<health::HealthState>>) -> Json<health::HealthState> {
    Json(state.as_ref().clone())
}
