use std::{path::Path, sync::Arc};

use axum::{extract::State, Json};
use ctranslate2_bindings::{
    TextInferenceEngine, TextInferenceEngineCreateOptionsBuilder, TextInferenceOptionsBuilder,
};
use serde::{Deserialize, Serialize};
use strfmt::{strfmt, strfmt_builder};
use tabby_common::{events, path::ModelDir};
use utoipa::ToSchema;

mod languages;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct CompletionRequest {
    #[schema(example = "def fib(n):")]
    #[deprecated]
    prompt: Option<String>,

    /// https://code.visualstudio.com/docs/languages/identifiers
    #[schema(example = "python")]
    language: Option<String>,

    /// When segments are set, the `prompt` is ignored during the inference.
    segments: Option<Segments>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Segments {
    /// Content that appears before the cursor in the editor window.
    #[schema(example = "def fib(n):\n    ")]
    prefix: String,

    /// Content that appears after the cursor in the editor window.
    #[schema(example = "\n        return fib(n - 1) + fib(n - 2)")]
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
    request_body = CompletionRequest ,
)]
pub async fn completion(
    State(state): State<Arc<CompletionState>>,
    Json(request): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let options = TextInferenceOptionsBuilder::default()
        .max_decoding_length(64)
        .sampling_temperature(0.2)
        .build()
        .expect("Invalid TextInferenceOptions");

    let prompt = if let Some(Segments { prefix, suffix }) = request.segments {
        if let Some(prompt_template) = &state.prompt_template {
            if let Some(suffix) = suffix {
                strfmt!(prompt_template, prefix => prefix, suffix => suffix)
                    .expect("Failed to format prompt")
            } else {
                // If suffix is empty, just returns prefix.
                prefix
            }
        } else {
            // If there's no prompt template, just use prefix.
            prefix
        }
    } else {
        request.prompt.expect("No prompt is set")
    };

    let text = state.engine.inference(&prompt, options).await;
    let language = request.language.unwrap_or("unknown".into());
    let filtered_text = languages::remove_stop_words(&language, &text);

    let response = CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        choices: vec![Choice {
            index: 0,
            text: filtered_text.to_string(),
        }],
    };

    events::Event::Completion {
        completion_id: &response.id,
        language: &language,
        prompt: &prompt,
        choices: vec![events::Choice {
            index: 0,
            text: filtered_text,
        }],
    }
    .log();

    Json(response)
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
        let options = TextInferenceEngineCreateOptionsBuilder::default()
            .model_path(model_dir.ctranslate2_dir())
            .tokenizer_path(model_dir.tokenizer_file())
            .device(device)
            .model_type(metadata.auto_model)
            .device_indices(args.device_indices.clone())
            .num_replicas_per_device(args.num_replicas_per_device)
            .build()
            .expect("Invalid TextInferenceEngineCreateOptions");
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
    serdeconv::from_json_file(model_dir.metadata_file()).expect("Invalid metadata")
}
