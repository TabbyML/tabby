mod prompt;

use std::sync::Arc;

use async_stream::stream;
use axum::{
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use axum_streams::StreamBodyAs;
use prompt::ChatPromptBuilder;
use serde::{Deserialize, Serialize};
use tabby_inference::{TextGeneration, TextGenerationOptions, TextGenerationOptionsBuilder};
use tracing::instrument;
use utoipa::ToSchema;

pub struct ChatState {
    engine: Arc<Box<dyn TextGeneration>>,
    prompt_builder: ChatPromptBuilder,
}

impl ChatState {
    pub fn new(engine: Arc<Box<dyn TextGeneration>>, prompt_template: String) -> Self {
        Self {
            engine,
            prompt_builder: ChatPromptBuilder::new(prompt_template),
        }
    }
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
#[schema(example=json!({
    "messages": [
        Message { role: "user".to_owned(), content: "What is tail recursion?".to_owned()},
        Message { role: "assistant".to_owned(), content: "It's a kind of optimization in compiler?".to_owned()},
        Message { role: "user".to_owned(), content: "Could you share more details?".to_owned()},
    ]
}))]
pub struct ChatCompletionRequest {
    messages: Vec<Message>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct ChatCompletionChunk {
    content: String,
}

#[utoipa::path(
    post,
    path = "/v1beta/chat/completions",
    request_body = ChatCompletionRequest,
    operation_id = "chat_completions",
    tag = "v1beta",
    responses(
        (status = 200, description = "Success", body = ChatCompletionChunk, content_type = "application/jsonstream"),
        (status = 405, description = "When chat model is not specified, the endpoint will returns 405 Method Not Allowed"),
    )
)]
#[instrument(skip(state, request))]
pub async fn completions(
    State(state): State<Arc<ChatState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let (prompt, options) = parse_request(&state, request);
    let s = stream! {
        for await content in state.engine.generate_stream(&prompt, options).await {
            yield ChatCompletionChunk { content }
        }
    };

    StreamBodyAs::json_nl(s).into_response()
}

fn parse_request(
    state: &Arc<ChatState>,
    request: ChatCompletionRequest,
) -> (String, TextGenerationOptions) {
    let mut builder = TextGenerationOptionsBuilder::default();

    builder
        .max_input_length(2048)
        .max_decoding_length(1920)
        .sampling_temperature(0.1);

    (
        state.prompt_builder.build(&request.messages),
        builder.build().unwrap(),
    )
}
