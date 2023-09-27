use std::sync::Arc;

use async_stream::stream;
use axum::{extract::State, response::IntoResponse, Json};
use axum_streams::StreamBodyAs;
use serde::{Deserialize, Serialize};
use tabby_inference::{TextGeneration, TextGenerationOptionsBuilder, TextGenerationOptions};
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
        example = "# Dijkstra'\''s shortest path algorithm in Python (4 spaces indentation) + complexity analysis:\n\ndef"
    )]
    prompt: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct GenerateResponse {
    text: String,
}

#[utoipa::path(
    post,
    path = "/v1/generate",
    request_body = GenerateRequest,
    operation_id = "generate",
    tag = "v1",
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
    path = "/v1/generate_stream",
    request_body = GenerateRequest,
    operation_id = "generate_stream",
    tag = "v1",
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

fn build_options(_request: &GenerateRequest) -> TextGenerationOptions {
    TextGenerationOptionsBuilder::default()
        .max_input_length(2048)
        .max_decoding_length(usize::MAX)
        .sampling_temperature(0.1)
        .build()
        .unwrap()
}
