use axum::Json;
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use tabby_common::events;
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct LogEventRequest {
    #[schema(example = "view")]
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
    if request.event_type == "view" {
        events::Event::View {
            completion_id: &request.completion_id,
            choice_index: request.choice_index,
        }
        .log();
        StatusCode::OK
    } else if request.event_type == "select" {
        events::Event::Select {
            completion_id: &request.completion_id,
            choice_index: request.choice_index,
        }
        .log();
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    }
}
