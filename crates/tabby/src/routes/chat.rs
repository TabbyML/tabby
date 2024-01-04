use std::sync::Arc;

use async_stream::stream;
use axum::{
    body::StreamBody,
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use tabby_common::api::event::Event::ChatCompletion;
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
    let prompt = Arc::new(request.prompt());
    let s = stream! {
        for await content in state.generate(&request).await {

            let id = content.id.clone();
            let content = match serde_json::to_string(&content) {
                Ok(s) => s,
                Err(e) => {
                    yield Err(e);
                    continue
                }
            };

            let content = format!("data: {}\n\n", content);
            let event = ChatCompletion { completion_id: id, prompt: prompt.clone(), content: content.clone() };
            state.logger.log(event);

            yield Ok(content)
        }
    };

    StreamBody::new(s).into_response()
}
