mod languages;
mod prompt;

use std::{path::Path, sync::Arc};

use axum::{extract::State, Json};
use ctranslate2_bindings::{CTranslate2Engine, CTranslate2EngineOptionsBuilder};
use hyper::StatusCode;
use llama_cpp_bindings::{LlamaEngine, LlamaEngineOptionsBuilder};
use serde::{Deserialize, Serialize};
use tabby_common::{config::Config, events, path::ModelDir};
use tabby_inference::{TextGeneration, TextGenerationOptionsBuilder};
use tracing::{debug, instrument};
use utoipa::ToSchema;

use self::languages::get_stop_words;
use super::Device;
use crate::fatal;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
#[schema(example=json!({
    "language": "python",
    "segments": {
        "prefix": "def fib(n):\n    ",
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
    let options = TextGenerationOptionsBuilder::default()
        .max_decoding_length(128)
        .sampling_temperature(0.1)
        .stop_words(get_stop_words(&language))
        .build()
        .unwrap();

    let segments = if let Some(segments) = request.segments {
        segments
    } else if let Some(prompt) = request.prompt {
        Segments {
            prefix: prompt,
            suffix: None,
        }
    } else {
        return Err(StatusCode::BAD_REQUEST);
    };

    debug!("PREFIX: {}, SUFFIX: {:?}", segments.prefix, segments.suffix);
    let prompt = state.prompt_builder.build(&language, segments);
    debug!("PROMPT: {}", prompt);
    let completion_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let text = state.engine.generate(&prompt, options).await;

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
    engine: Box<dyn TextGeneration>,
    prompt_builder: prompt::PromptBuilder,
}

impl CompletionState {
    pub fn new(args: &crate::serve::ServeArgs, config: &Config) -> Self {
        let model_dir = get_model_dir(&args.model);
        let metadata = read_metadata(&model_dir);

        let engine: Box<dyn TextGeneration> = if args.device != Device::Metal {
            let device = format!("{}", args.device);
            let compute_type = format!("{}", args.compute_type);
            let options = CTranslate2EngineOptionsBuilder::default()
                .model_path(model_dir.ctranslate2_dir())
                .tokenizer_path(model_dir.tokenizer_file())
                .device(device)
                .model_type(metadata.auto_model)
                .device_indices(args.device_indices.clone())
                .num_replicas_per_device(args.num_replicas_per_device)
                .compute_type(compute_type)
                .build()
                .unwrap();
            Box::new(CTranslate2Engine::create(options))
        } else {
            let options = LlamaEngineOptionsBuilder::default()
                .model_path(model_dir.ggml_model_file())
                .tokenizer_path(model_dir.tokenizer_file())
                .build()
                .unwrap();

            Box::new(LlamaEngine::create(options))
        };

        Self {
            engine,
            prompt_builder: prompt::PromptBuilder::new(
                metadata.prompt_template,
                config.experimental.enable_prompt_rewrite,
            ),
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
