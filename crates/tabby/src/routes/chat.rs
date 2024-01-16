use std::sync::Arc;


use axum::{
    body::StreamBody,
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use futures::StreamExt;

use tracing::instrument;

use crate::services::chat::{ChatCompletionRequest, ChatService};

#[utoipa::path(
    post,
    path = "/v1beta/chat/completions",
    request_body = ChatCompletionRequest,
    operation_id = "chat_completions",
    tag = "v1beta",
    responses(
        (status = 200, description = "Success", body = ChatCompletionChunk, content_type = "text/event-stream"),
        (status = 405, description = "When chat model is not specified, the endpoint will returns 405 Method Not Allowed"),
    ),
    security(
        ("token" = [])
    )
)]
#[instrument(skip(state, request))]
pub async fn chat_completions(
    State(state): State<Arc<ChatService>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let stream = state.generate(request).await;
    let stream = match stream {
        Ok(s) => s,
        Err(_) => {
            let mut response = StreamBody::default().into_response();
            *response.status_mut() = hyper::StatusCode::UNPROCESSABLE_ENTITY;
            return response;
        }
    };
    let s = stream.map(|chunk| match serde_json::to_string(&chunk) {
        Ok(s) => Ok(format!("data: {s}")),
        Err(e) => Err(anyhow::Error::from(e)),
    });
    StreamBody::new(s).into_response()
}
