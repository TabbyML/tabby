use axum::Json;
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use tabby_common::events;
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct LogEventRequest {
    #[serde(rename = "type")]
    event_type: String,
    completion_id: String,
    choice_index: u32,
}

#[derive(Serialize)]
struct LogEvent {
    completion_id: String,
    choice_index: u32,
}

#[utoipa::path(
    post,
    path = "/v1/events",
    request_body = LogEventRequest,
)]
pub async fn log_event(Json(request): Json<LogEventRequest>) -> StatusCode {
    events::log(
        &request.event_type,
        &LogEvent {
            completion_id: request.completion_id,
            choice_index: request.choice_index,
        },
    );
    StatusCode::OK
}
