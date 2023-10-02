use std::sync::Arc;

use async_stream::stream;
use axum::{extract::State, response::IntoResponse, Json};
use axum_streams::StreamBodyAs;
use serde::{Deserialize, Serialize};
use tabby_inference::{TextGeneration, TextGenerationOptions, TextGenerationOptionsBuilder};
use tracing::instrument;
use utoipa::ToSchema;

pub struct ChatState {
    engine: Arc<Box<dyn TextGeneration>>,
}

impl ChatState {
    pub fn new(engine: Arc<Box<dyn TextGeneration>>) -> Self {
        Self { engine }
    }
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
#[schema(example=json!({
    "prompt": "# Dijkstra'\''s shortest path algorithm in Python (4 spaces indentation) + complexity analysis:\n\n",
}))]
pub struct ChatCompletionRequest {
    prompt: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct ChatCompletionResponse {
    text: String,
}

#[utoipa::path(
    post,
    path = "/v1beta/chat/completions",
    request_body = ChatCompletionRequest,
    operation_id = "chat_completions",
    tag = "v1beta",
    responses(
        (status = 200, description = "Success", body = ChatCompletionResponse, content_type = "application/jsonstream"),
    )
)]
#[instrument(skip(state, request))]
pub async fn completions(
    State(state): State<Arc<ChatState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let (prompt, options) = parse_request(request);
    let s = stream! {
        for await text in state.engine.generate_stream(&prompt, options).await {
            yield ChatCompletionResponse { text }
        }
    };

    StreamBodyAs::json_nl(s)
}

fn parse_request(request: ChatCompletionRequest) -> (String, TextGenerationOptions) {
    let mut builder = TextGenerationOptionsBuilder::default();

    builder
        .max_input_length(1024)
        .max_decoding_length(968)
        .sampling_temperature(0.1);

    (request.prompt, builder.build().unwrap())
}
