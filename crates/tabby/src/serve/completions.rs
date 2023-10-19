mod prompt;

use std::sync::Arc;

use axum::{extract::State, Json};
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use tabby_common::{events, languages::get_language};
use tabby_inference::{TextGeneration, TextGenerationOptionsBuilder};
use tracing::{debug, instrument};
use utoipa::ToSchema;

use super::search::IndexServer;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
#[schema(example=json!({
    "language": "python",
    "segments": {
        "prefix": "def fib(n):\n    ",
        "suffix": "\n        return fib(n - 1) + fib(n - 2)"
    }
}))]
pub struct CompletionRequest {
    /// Language identifier, full list is maintained at
    /// https://code.visualstudio.com/docs/languages/identifiers
    #[schema(example = "python")]
    language: Option<String>,

    /// When segments are set, the `prompt` is ignored during the inference.
    segments: Option<Segments>,

    /// When prompt is specified, it will be passed directly to the inference engine for completion.
    /// This is useful for certain requests that aim to test the model's quality.
    prompt: Option<String>,

    /// A unique identifier representing your end-user, which can help Tabby to monitor & generating
    /// reports.
    user: Option<String>,

    debug_options: Option<DebugOptions>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct DebugOptions {
    /// When true, returns debug_data in completion response.
    #[serde(default = "default_false")]
    enabled: bool,

    /// When true, disable retrieval augmented code completion.
    #[serde(default = "default_false")]
    disable_retrieval_augmented_code_completion: bool,
}

fn default_false() -> bool {
    false
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
pub struct Snippet {
    filepath: String,
    body: String,
    score: f32,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
#[schema(example=json!({
    "id": "string",
    "choices": [ { "index": 0, "text": "string" } ]
}))]
pub struct CompletionResponse {
    id: String,
    choices: Vec<Choice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    debug_data: Option<DebugData>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct DebugData {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    snippets: Vec<Snippet>,

    prompt: String,
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
pub async fn completions(
    State(state): State<Arc<CompletionState>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    let language = request.language.unwrap_or("unknown".to_string());
    let options = TextGenerationOptionsBuilder::default()
        .max_input_length(1024 + 512)
        .max_decoding_length(128)
        .sampling_temperature(0.1)
        .language(get_language(&language))
        .build()
        .unwrap();

    let (prompt, segments, snippets) = if let Some(segments) = request.segments {
        debug!("PREFIX: {}, SUFFIX: {:?}", segments.prefix, segments.suffix);
        let (prompt, snippets) = build_prompt(&state, &request.debug_options, &language, &segments);
        (prompt, Some(segments), snippets)
    } else if let Some(prompt) = request.prompt {
        (prompt, None, vec![])
    } else {
        return Err(StatusCode::BAD_REQUEST);
    };
    debug!("PROMPT: {}", prompt);

    let completion_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let text = state.engine.generate(&prompt, options).await;

    let segments = segments.map(|x| tabby_common::events::Segments {
        prefix: x.prefix,
        suffix: x.suffix,
    });

    events::Event::Completion {
        completion_id: &completion_id,
        language: &language,
        prompt: &prompt,
        segments: &segments,
        choices: vec![events::Choice {
            index: 0,
            text: &text,
        }],
        user: request.user.as_deref(),
    }
    .log();

    let debug_data = DebugData { snippets, prompt };

    Ok(Json(CompletionResponse {
        id: completion_id,
        choices: vec![Choice { index: 0, text }],
        debug_data: if request.debug_options.is_some_and(|x| x.enabled) {
            Some(debug_data)
        } else {
            None
        },
    }))
}

fn build_prompt(
    state: &Arc<CompletionState>,
    debug_options: &Option<DebugOptions>,
    language: &str,
    segments: &Segments,
) -> (String, Vec<Snippet>) {
    let snippets = if !debug_options
        .as_ref()
        .is_some_and(|x| x.disable_retrieval_augmented_code_completion)
    {
        state.prompt_builder.collect(&language, &segments)
    } else {
        vec![]
    };
    (
        state
            .prompt_builder
            .build(&language, segments.clone(), &snippets),
        snippets,
    )
}

pub struct CompletionState {
    engine: Arc<Box<dyn TextGeneration>>,
    prompt_builder: prompt::PromptBuilder,
}

impl CompletionState {
    pub fn new(
        engine: Arc<Box<dyn TextGeneration>>,
        index_server: Arc<IndexServer>,
        prompt_template: Option<String>,
    ) -> Self {
        Self {
            engine,
            prompt_builder: prompt::PromptBuilder::new(prompt_template, Some(index_server)),
        }
    }
}
