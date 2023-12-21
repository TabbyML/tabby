use std::{convert::Infallible, sync::Arc};

use async_stream::stream;
use axum::{
    body::{BoxBody, StreamBody},
    extract::State,
    http::HeaderValue,
    response::Response,
    Json,
};
use tracing::instrument;
use uuid::Uuid;

use crate::services::chat::{format_chunk_to_event, ChatCompletionRequest, ChatService};

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
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("System clock must be accessible")
        .as_secs();
    let uuid = Uuid::new_v4().to_string();
    let s = stream! {
        for await content in state.generate(&request).await {
            let content = format_chunk_to_event(content, &uuid, time, false);
            yield Ok::<_, Infallible>(content);
        }
        yield Ok(format_chunk_to_event(Default::default(), &uuid, time, true));
    };

    let body = BoxBody::new(StreamBody::new(s));
    let mut res = Response::new(body);
    res.headers_mut().insert(
        "Content-Type",
        HeaderValue::from_static("text/event-stream"),
    );
    res
}
