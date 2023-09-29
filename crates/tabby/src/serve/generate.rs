use std::sync::Arc;

use async_stream::stream;
use axum::{extract::State, response::IntoResponse, Json};
use axum_streams::StreamBodyAs;
use serde::{Deserialize, Serialize};
use tabby_inference::{TextGeneration, TextGenerationOptions, TextGenerationOptionsBuilder};
use tracing::instrument;
use utoipa::ToSchema;

pub struct GenerateState {
    engine: Arc<Box<dyn TextGeneration>>,
}

impl GenerateState {
    pub fn new(engine: Arc<Box<dyn TextGeneration>>) -> Self {
        Self { engine }
    }
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
#[schema(example=json!({
    "prompt": "# Dijkstra'\''s shortest path algorithm in Python (4 spaces indentation) + complexity analysis:\n\n",
}))]
pub struct GenerateRequest {
    prompt: String,
    stop_words: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct GenerateResponse {
    text: String,
}

#[utoipa::path(
    post,
    path = "/v1beta/generate",
    request_body = GenerateRequest,
    operation_id = "generate",
    tag = "v1beta",
    responses(
        (status = 200, description = "Success", body = GenerateResponse, content_type = "application/json"),
    )
)]
#[instrument(skip(state, request))]
pub async fn generate(
    State(state): State<Arc<GenerateState>>,
    Json(request): Json<GenerateRequest>,
) -> impl IntoResponse {
    let (prompt, options) = parse_request(request);
    Json(GenerateResponse {
        text: state.engine.generate(&prompt, options).await,
    })
}

#[utoipa::path(
    post,
    path = "/v1beta/generate_stream",
    request_body = GenerateRequest,
    operation_id = "generate_stream",
    tag = "v1beta",
    responses(
        (status = 200, description = "Success", body = GenerateResponse, content_type = "application/jsonstream"),
    )
)]
#[instrument(skip(state, request))]
pub async fn generate_stream(
    State(state): State<Arc<GenerateState>>,
    Json(request): Json<GenerateRequest>,
) -> impl IntoResponse {
    let (prompt, options) = parse_request(request);
    let s = stream! {
        for await text in state.engine.generate_stream(&prompt, options).await {
            yield GenerateResponse { text }
        }
    };

    StreamBodyAs::json_nl(s)
}

fn parse_request(request: GenerateRequest) -> (String, TextGenerationOptions) {
    let mut builder = TextGenerationOptionsBuilder::default();

    builder
        .max_input_length(1024)
        .max_decoding_length(968)
        .sampling_temperature(0.1);

    if let Some(stop_words) = request.stop_words {
        builder.stop_words(stop_words);
    };

    (request.prompt, builder.build().unwrap())
}
