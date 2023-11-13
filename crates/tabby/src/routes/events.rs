use std::collections::HashMap;

use axum::{extract::Query, Json};
use hyper::StatusCode;

use tabby_common::events::{self, SelectKind};
use utoipa::ToSchema;

use crate::api::LogEventRequest;

#[utoipa::path(
    post,
    path = "/v1/events",
    request_body = LogEventRequest,
    tag = "v1",
    operation_id = "event",
    responses(
        (status = 200, description = "Success"),
        (status = 400, description = "Bad Request")
    )
)]
pub async fn log_event(
    Query(params): Query<HashMap<String, String>>,
    Json(request): Json<LogEventRequest>,
) -> StatusCode {
    if request.event_type == "view" {
        events::Event::View {
            completion_id: &request.completion_id,
            choice_index: request.choice_index,
        }
        .log();
        StatusCode::OK
    } else if request.event_type == "select" {
        let is_line = params
            .get("select_kind")
            .map(|x| x == "line")
            .unwrap_or(false);
        events::Event::Select {
            completion_id: &request.completion_id,
            choice_index: request.choice_index,
            kind: if is_line {
                Some(SelectKind::Line)
            } else {
                None
            },
        }
        .log();
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    }
}
