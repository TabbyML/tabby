use std::sync::Arc;

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use axum_extra::TypedHeader;
use futures::{Stream, StreamExt};
use hyper::StatusCode;
use tabby_common::axum::MaybeUser;
use tabby_inference::ChatCompletionStream;
use tracing::{instrument, warn};

#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    operation_id = "chat_completions",
    tag = "v1",
    responses(
        (status = 200, description = "Success", content_type = "text/event-stream"),
        (status = 405, description = "When chat model is not specified, the endpoint returns 405 Method Not Allowed"),
        (status = 422, description = "When the prompt is malformed, the endpoint returns 422 Unprocessable Entity")
    ),
    security(
        ("token" = [])
    )
)]
pub async fn chat_completions_utoipa(_request: Json<serde_json::Value>) -> StatusCode {
    unimplemented!()
}

#[instrument(skip(state, request))]
pub async fn chat_completions(
    State(state): State<Arc<dyn ChatCompletionStream>>,
    TypedHeader(MaybeUser(user)): TypedHeader<MaybeUser>,
    Json(mut request): Json<async_openai::types::CreateChatCompletionRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, anyhow::Error>>>, StatusCode> {
    if let Some(user) = user {
        request.user.replace(user);
    }

    let s = match state.chat_stream(request).await {
        Ok(s) => s,
        Err(err) => {
            warn!("Error happens during chat completion: {}", err);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let s = s.map(|chunk| {
        let chunk = chunk?;
        let json = serde_json::to_string(&chunk)?;
        Ok(Event::default().data(json))
    });

    Ok(Sse::new(s).keep_alive(KeepAlive::default()))
}
