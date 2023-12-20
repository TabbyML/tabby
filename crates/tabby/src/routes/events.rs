use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Query, State},
    Json,
};
use hyper::StatusCode;
use tabby_common::api::event::{Event, EventLogger, LogEventRequest, SelectKind};

#[utoipa::path(
    post,
    path = "/v1/events",
    request_body = LogEventRequest,
    tag = "v1",
    operation_id = "event",
    responses(
        (status = 200, description = "Success"),
        (status = 400, description = "Bad Request")
    ),
    security(
        ("token" = [])
    )
)]
pub async fn log_event(
    State(logger): State<Arc<dyn EventLogger>>,
    Query(params): Query<HashMap<String, String>>,
    Json(request): Json<LogEventRequest>,
) -> StatusCode {
    if request.event_type == "view" {
        logger.log(Event::View {
            completion_id: request.completion_id,
            choice_index: request.choice_index,
        });
        StatusCode::OK
    } else if request.event_type == "select" {
        let is_line = params
            .get("select_kind")
            .map(|x| x == "line")
            .unwrap_or(false);
        logger.log(Event::Select {
            completion_id: request.completion_id,
            choice_index: request.choice_index,
            kind: if is_line {
                Some(SelectKind::Line)
            } else {
                None
            },
        });
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    }
}
