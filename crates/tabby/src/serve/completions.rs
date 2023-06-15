use std::{path::Path, sync::Arc};

use axum::{extract::State, Json};
use ctranslate2_bindings::{
    TextInferenceEngine, TextInferenceEngineCreateOptionsBuilder, TextInferenceOptionsBuilder,
};
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use strfmt::{strfmt, strfmt_builder};
use tabby_common::{events, path::ModelDir};
use tracing::instrument;
use utoipa::ToSchema;

use self::languages::get_stop_words;
use crate::fatal;

mod languages;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
#[schema(example=json!({
    "language": "python",
    "segments": {
        "prefix": "def fib(n)\n    ",
        "suffix": "\n        return fib(n - 1) + fib(n - 2)"
    }
}))]
pub struct CompletionRequest {
    #[schema(example = "def fib(n):")]
    prompt: Option<String>,

    /// Language identifier, full list is maintained at
    /// https://code.visualstudio.com/docs/languages/identifiers
    #[schema(example = "python")]
    language: Option<String>,

    /// When segments are set, the `prompt` is ignored during the inference.
    segments: Option<Segments>,

    // A unique identifier representing your end-user, which can help Tabby to monitor & generating
    // reports.
    user: Option<String>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Segments {
    /// Content that appears before the cursor in the editor window.
    prefix: String,

    /// Content that appears after the cursor in the editor window.
    suffix: Option<String>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Choice {
    index: u32,
    text: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct CompletionResponse {
    id: String,
    choices: Vec<Choice>,
}

#[utoipa::path(
    post,
    path = "/v1/completions",
    request_body = CompletionRequest,
    operation_id = "completion",
    tag = "v1",
    responses(
        (status = 200, description = "Success", body = CompletionResponse, content_type = "application/json"),
        (status = 400, description = "Bad Request")
    )
)]
#[instrument(skip(state, request))]
pub async fn completion(
    State(state): State<Arc<CompletionState>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    let language = request.language.unwrap_or("unknown".to_string());
    let options = TextInferenceOptionsBuilder::default()
        .max_decoding_length(128)
        .sampling_temperature(0.1)
        .stop_words(get_stop_words(&language))
        .build()
        .unwrap();

    let prompt = if let Some(Segments { prefix, suffix }) = request.segments {
        if let (Some(prompt_template), Some(suffix)) = (&state.prompt_template, suffix) {
            if !suffix.is_empty() {
                strfmt!(prompt_template, prefix => prefix, suffix => suffix).unwrap()
            } else {
                prefix
            }
        } else {
            // If there's no prompt template, just use prefix.
            prefix
        }
    } else if let Some(prompt) = request.prompt {
        prompt
    } else {
        return Err(StatusCode::BAD_REQUEST);
    };

    let completion_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let text = state.engine.inference(&prompt, options).await;

    events::Event::Completion {
        completion_id: &completion_id,
        language: &language,
        prompt: &prompt,
        choices: vec![events::Choice {
            index: 0,
            text: &text,
        }],
        user: request.user.as_deref(),
    }
    .log();

    Ok(Json(CompletionResponse {
        id: completion_id,
        choices: vec![Choice { index: 0, text }],
    }))
}

pub struct CompletionState {
    engine: TextInferenceEngine,
    prompt_template: Option<String>,
}

impl CompletionState {
    pub fn new(args: &crate::serve::ServeArgs) -> Self {
        let model_dir = get_model_dir(&args.model);
        let metadata = read_metadata(&model_dir);

        let device = format!("{}", args.device);
        let compute_type = format!("{}", args.compute_type);
        let options = TextInferenceEngineCreateOptionsBuilder::default()
            .model_path(model_dir.ctranslate2_dir())
            .tokenizer_path(model_dir.tokenizer_file())
            .device(device)
            .model_type(metadata.auto_model)
            .device_indices(args.device_indices.clone())
            .num_replicas_per_device(args.num_replicas_per_device)
            .compute_type(compute_type)
            .build()
            .unwrap();
        let engine = TextInferenceEngine::create(options);
        Self {
            engine,
            prompt_template: metadata.prompt_template,
        }
    }
}

fn get_model_dir(model: &str) -> ModelDir {
    if Path::new(model).exists() {
        ModelDir::from(model)
    } else {
        ModelDir::new(model)
    }
}

#[derive(Deserialize)]
struct Metadata {
    auto_model: String,
    prompt_template: Option<String>,
}

fn read_metadata(model_dir: &ModelDir) -> Metadata {
    serdeconv::from_json_file(model_dir.metadata_file())
        .unwrap_or_else(|_| fatal!("Invalid metadata file: {}", model_dir.metadata_file()))
}
