use std::sync::Arc;

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct HealthState {
    model: String,
    device: String,
    compute_type: String,
}

impl HealthState {
    pub fn new(args: &super::ServeArgs) -> Self {
        Self {
            model: args.model.clone(),
            device: args.device.to_string(),
            compute_type: args.compute_type.to_string(),
        }
    }
}

#[utoipa::path(
    post,
    path = "/v1/health",
    tag = "v1",
    responses(
        (status = 200, description = "Success", body = HealthState, content_type = "application/json"),
    )
)]
pub async fn health(State(state): State<Arc<HealthState>>) -> Json<HealthState> {
    Json(state.as_ref().clone())
}
