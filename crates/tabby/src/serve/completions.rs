use axum::{extract::State, Json};
use ctranslate2_bindings::{
    TextInferenceEngine, TextInferenceEngineCreateOptions, TextInferenceOptionsBuilder,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{trace, span, Level};
use utoipa::ToSchema;

mod languages;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct CompletionRequest {
    /// https://code.visualstudio.com/docs/languages/identifiers
    #[schema(example = "python")]
    language: String,

    #[schema(example = "def fib(n):")]
    prompt: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Choice {
    index: u32,
    text: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct CompletionResponse {
    id: String,
    created: u64,
    choices: Vec<Choice>,
}

#[utoipa::path(
    post,
    path = "/v1/completions",
    request_body = CompletionRequest ,
)]
pub async fn completion(
    State(state): State<Arc<CompletionState>>,
    Json(request): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let span = span!(Level::TRACE, "completion");
    let _enter = span.enter();

    trace!(language = request.language, prompt = request.prompt);
    let options = TextInferenceOptionsBuilder::default()
        .max_decoding_length(64)
        .sampling_temperature(0.2)
        .build()
        .unwrap();
    let text = state.engine.inference(&request.prompt, options);
    let filtered_text = languages::remove_stop_words(&request.language, &text);
    trace!(response = filtered_text);
    Json(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        created: timestamp(),
        choices: [Choice {
            index: 0,
            text: filtered_text.to_string(),
        }]
        .to_vec(),
    })
}

pub struct CompletionState {
    engine: TextInferenceEngine,
}

impl CompletionState {
    pub fn new(options: TextInferenceEngineCreateOptions) -> Self {
        let engine = TextInferenceEngine::create(options);
        Self { engine }
    }
}

fn timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let start = SystemTime::now();
    start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}
