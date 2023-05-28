use axum::Json;
use tracing::{span, info, Level};
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct LogEventRequest {
    #[serde(rename = "type")]
    event_type: String,
    completion_id: String,
    choice_index: u32,
}

#[utoipa::path(
    post,
    path = "/v1/events",
    request_body = LogEventRequest,
)]
pub async fn log_event(Json(request): Json<LogEventRequest>) -> StatusCode {
    info!(
        completion_id=request.completion_id,
        event_type=request.event_type,
        choice_index=request.choice_index
    );
    StatusCode::OK
}
