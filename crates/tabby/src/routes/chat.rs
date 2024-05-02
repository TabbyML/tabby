use std::{convert::Infallible, sync::Arc, time::Duration};

use axum::{
    extract::State,
    http::HeaderValue,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures::{stream, Stream, StreamExt};
use tracing::instrument;

use crate::services::chat::{ChatCompletionRequest, ChatService};

#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    request_body = ChatCompletionRequest,
    operation_id = "chat_completions",
    tag = "v1",
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
) -> Sse<impl Stream<Item = Result<Event, serde_json::Error>>> {
    let stream = state.generate(request).await;
    Sse::new(stream.map(|chunk| match serde_json::to_string(&chunk) {
        Ok(s) => Ok(Event::default().data(s)),
        Err(err) => Err(err),
    }))
    .keep_alive(KeepAlive::default())
}
