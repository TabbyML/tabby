use std::sync::Arc;

use axum::{extract::State, Json};
use hyper::StatusCode;
use tracing::{instrument, warn};

use crate::services::completion::{CompletionRequest, CompletionResponse, CompletionService};

#[utoipa::path(
    post,
    path = "/v1/completions",
    request_body = CompletionRequest,
    operation_id = "completion",
    tag = "v1",
    responses(
        (status = 200, description = "Success", body = CompletionResponse, content_type = "application/json"),
        (status = 400, description = "Bad Request")
    ),
    security(
        ("token" = [])
    )
)]
#[instrument(skip(state, request))]
pub async fn completions(
    State(state): State<Arc<CompletionService>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    match state.generate(&request).await {
        Ok(resp) => Ok(Json(resp)),
        Err(err) => {
            warn!("{}", err);
            Err(StatusCode::BAD_REQUEST)
        }
    }
}
