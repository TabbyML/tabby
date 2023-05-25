use axum::Json;
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
    println!("log_event: {:?}", request);
    StatusCode::OK
}
