use std::sync::Arc;

use async_openai_alt::error::OpenAIError;
use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use axum_extra::TypedHeader;
use futures::{Stream, StreamExt};
use hyper::StatusCode;
use tabby_common::{
    api::event::{Event as LoggerEvent, EventLogger},
    axum::MaybeUser,
};
use tabby_inference::ChatCompletionStream;
use tracing::{error, instrument, warn};

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
#[allow(unused)]
pub async fn chat_completions_utoipa(_request: Json<serde_json::Value>) -> StatusCode {
    unimplemented!()
}

pub struct ChatState {
    pub chat_completion: Arc<dyn ChatCompletionStream>,
    pub logger: Arc<dyn EventLogger>,
}

#[instrument(skip(state, request))]
pub async fn chat_completions(
    State(state): State<Arc<ChatState>>,
    TypedHeader(MaybeUser(user)): TypedHeader<MaybeUser>,
    Json(mut request): Json<async_openai_alt::types::CreateChatCompletionRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, anyhow::Error>>>, StatusCode> {
    if let Some(user) = user {
        request.user.replace(user);
    }
    let user = request.user.clone();

    let s = match state.chat_completion.chat_stream(request).await {
        Ok(s) => s,
        Err(err) => {
            warn!("Error happens during chat completion: {}", err);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let s = async_stream::stream! {
        let mut s = s;
        while let Some(event) = s.next().await {
            match event {
                Ok(event) => {
                    yield Ok(Event::default().json_data(event)?);
                }
                Err(err) => {
                    if let OpenAIError::StreamError(content) = err {
                        if content == "Stream ended" {
                            break;
                        }
                    } else {
                        error!("Failed to get chat completion chunk: {:?}", err);
                        yield Err(err.into());
                    }
                }
            }
        }
    };

    state.logger.log(user, LoggerEvent::ChatCompletion {});

    Ok(Sse::new(s).keep_alive(KeepAlive::default()))
}
