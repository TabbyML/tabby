use std::sync::Arc;

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use axum_extra::TypedHeader;
use futures::{Stream, StreamExt};
use tracing::instrument;

use super::MaybeUser;
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
    TypedHeader(MaybeUser(user)): TypedHeader<MaybeUser>,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event, serde_json::Error>>> {
    if let Some(user) = user {
        request.user.replace(user);
    }

    let stream = state.generate(request).await;
    Sse::new(stream.map(|chunk| match serde_json::to_string(&chunk) {
        Ok(s) => Ok(Event::default().data(s)),
        Err(err) => Err(err),
    }))
    .keep_alive(KeepAlive::default())
}
