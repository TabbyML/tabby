use std::sync::Arc;

use async_stream::stream;
use axum::{
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use axum_streams::StreamBodyAs;
use tracing::instrument;

use crate::services::chat::{ChatCompletionRequest, ChatService};

#[utoipa::path(
    post,
    path = "/v1beta/chat/completions",
    request_body = ChatCompletionRequest,
    operation_id = "chat_completions",
    tag = "v1beta",
    responses(
        (status = 200, description = "Success", body = ChatCompletionChunk, content_type = "application/jsonstream"),
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
    let s = stream! {
        for await content in state.generate(&request).await {
            yield content;
        }
    };

    StreamBodyAs::json_nl(s).into_response()
}
