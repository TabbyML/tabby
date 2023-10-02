use std::sync::Arc;

use async_stream::stream;
use axum::{extract::State, response::IntoResponse, Json};
use axum_streams::StreamBodyAs;
use lazy_static::lazy_static;
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
pub struct GenerateRequest {
    #[schema(
        example = "# Dijkstra'\''s shortest path algorithm in Python (4 spaces indentation) + complexity analysis:\n\n"
    )]
    prompt: String,
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
    let options = build_options(&request);
    Json(GenerateResponse {
        text: state.engine.generate(&request.prompt, options).await,
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
    let options = build_options(&request);
    let s = stream! {
        for await text in state.engine.generate_stream(&request.prompt, options).await {
            yield GenerateResponse { text }
        }
    };

    StreamBodyAs::json_nl(s)
}

lazy_static! {
    static ref STOP_WORDS: Vec<&'static str> = vec!["\n\n",];
}

fn build_options(_request: &GenerateRequest) -> TextGenerationOptions {
    TextGenerationOptionsBuilder::default()
        .max_input_length(1024)
        .max_decoding_length(1024)
        .sampling_temperature(0.1)
        .stop_words(&STOP_WORDS)
        .build()
        .unwrap()
}
