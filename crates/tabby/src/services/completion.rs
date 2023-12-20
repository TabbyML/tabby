mod completion_prompt;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tabby_common::{
    api,
    api::{
        code::CodeSearch,
        event::{Event, EventLogger},
    },
    languages::get_language,
};
use tabby_inference::{TextGeneration, TextGenerationOptions, TextGenerationOptionsBuilder};
use thiserror::Error;
use tracing::debug;
use utoipa::ToSchema;

use super::model;
use crate::Device;

#[derive(Error, Debug)]
pub enum CompletionError {
    #[error("empty prompt from completion request")]
    EmptyPrompt,
}

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

    /// A unique identifier representing your end-user, which can help Tabby to monitor & generating
    /// reports.
    user: Option<String>,

    debug_options: Option<DebugOptions>,
}

impl CompletionRequest {
    /// Returns the language info or "unknown" if not specified.
    fn language_or_unknown(&self) -> String {
        self.language.clone().unwrap_or("unknown".to_string())
    }

    /// Returns the raw prompt if specified.
    fn raw_prompt(&self) -> Option<String> {
        self.debug_options
            .as_ref()
            .and_then(|x| x.raw_prompt.clone())
    }

    /// Returns true if retrieval augmented code completion is disabled.
    fn disable_retrieval_augmented_code_completion(&self) -> bool {
        self.debug_options
            .as_ref()
            .is_some_and(|x| x.disable_retrieval_augmented_code_completion)
    }
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct DebugOptions {
    /// When `raw_prompt` is specified, it will be passed directly to the inference engine for completion. `segments` field in `CompletionRequest` will be ignored.
    ///
    /// This is useful for certain requests that aim to test the tabby's e2e quality.
    raw_prompt: Option<String>,

    /// When true, returns `snippets` in `debug_data`.
    #[serde(default = "default_false")]
    return_snippets: bool,

    /// When true, returns `prompt` in `debug_data`.
    #[serde(default = "default_false")]
    return_prompt: bool,

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

    /// Clipboard content when requesting code completion.
    clipboard: Option<String>,
}

impl From<Segments> for api::event::Segments {
    fn from(val: Segments) -> Self {
        Self {
            prefix: val.prefix,
            suffix: val.suffix,
            clipboard: val.clipboard,
        }
    }
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Choice {
    index: u32,
    text: String,
}

impl Choice {
    pub fn new(text: String) -> Self {
        Self { index: 0, text }
    }
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

impl CompletionResponse {
    pub fn new(id: String, choices: Vec<Choice>, debug_data: Option<DebugData>) -> Self {
        Self {
            id,
            choices,
            debug_data,
        }
    }
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct DebugData {
    #[serde(skip_serializing_if = "Option::is_none")]
    snippets: Option<Vec<Snippet>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<String>,
}

pub struct CompletionService {
    engine: Arc<dyn TextGeneration>,
    logger: Arc<dyn EventLogger>,
    prompt_builder: completion_prompt::PromptBuilder,
}

impl CompletionService {
    fn new(
        engine: Arc<dyn TextGeneration>,
        code: Arc<dyn CodeSearch>,
        logger: Arc<dyn EventLogger>,
        prompt_template: Option<String>,
    ) -> Self {
        Self {
            engine,
            prompt_builder: completion_prompt::PromptBuilder::new(prompt_template, Some(code)),
            logger,
        }
    }

    async fn build_snippets(
        &self,
        language: &str,
        segments: &Segments,
        disable_retrieval_augmented_code_completion: bool,
    ) -> Vec<Snippet> {
        if !disable_retrieval_augmented_code_completion {
            self.prompt_builder.collect(language, segments).await
        } else {
            vec![]
        }
    }

    fn text_generation_options(language: &str) -> TextGenerationOptions {
        TextGenerationOptionsBuilder::default()
            .max_input_length(1024 + 512)
            .max_decoding_length(128)
            .sampling_temperature(0.1)
            .language(Some(get_language(language)))
            .build()
            .unwrap()
    }

    pub async fn generate(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let completion_id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let language = request.language_or_unknown();
        let options = Self::text_generation_options(language.as_str());

        let (prompt, segments, snippets) = if let Some(prompt) = request.raw_prompt() {
            (prompt, None, vec![])
        } else if let Some(segments) = request.segments.clone() {
            debug!("PREFIX: {}, SUFFIX: {:?}", segments.prefix, segments.suffix);
            let snippets = self
                .build_snippets(
                    &language,
                    &segments,
                    request.disable_retrieval_augmented_code_completion(),
                )
                .await;
            let prompt = self
                .prompt_builder
                .build(&language, segments.clone(), &snippets);
            (prompt, Some(segments), snippets)
        } else {
            return Err(CompletionError::EmptyPrompt);
        };
        debug!("PROMPT: {}", prompt);

        let text = self.engine.generate(&prompt, options).await;
        let segments = segments.map(|s| s.into());

        self.logger.log(Event::Completion {
            completion_id: completion_id.clone(),
            language,
            prompt: prompt.clone(),
            segments,
            choices: vec![api::event::Choice {
                index: 0,
                text: text.clone(),
            }],
            user: request.user.clone(),
        });

        let debug_data = request
            .debug_options
            .as_ref()
            .map(|debug_options| DebugData {
                snippets: debug_options.return_snippets.then_some(snippets),
                prompt: debug_options.return_prompt.then_some(prompt),
            });

        Ok(CompletionResponse::new(
            completion_id,
            vec![Choice::new(text)],
            debug_data,
        ))
    }
}

pub async fn create_completion_service(
    code: Arc<dyn CodeSearch>,
    logger: Arc<dyn EventLogger>,
    model: &str,
    device: &Device,
    parallelism: u8,
) -> CompletionService {
    let (
        engine,
        model::PromptInfo {
            prompt_template, ..
        },
    ) = model::load_text_generation(model, device, parallelism).await;

    CompletionService::new(engine.clone(), code, logger, prompt_template)
}
