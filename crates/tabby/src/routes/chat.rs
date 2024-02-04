use std::sync::Arc;

use axum::{
    body::StreamBody,
    extract::State,
    http::HeaderValue,
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
        (status = 405, description = "When chat model is not specified, the endpoint returns 405 Method Not Allowed"),
        (status = 422, description = "When the prompt is malformed, the endpoint returns 422 Unprocessable Entity")
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
        Ok(s) => Ok(format!("data: {s}\n\n")),
        Err(e) => Err(anyhow::Error::from(e)),
    });
    let mut resp = StreamBody::new(s).into_response();
    resp.headers_mut().append(
        "Content-Type",
        HeaderValue::from_str("text/event-stream").unwrap(),
    );
    resp
}
