mod chat_prompt;

use std::sync::Arc;

use async_stream::stream;
use chat_prompt::ChatPromptBuilder;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use tabby_inference::{TextGeneration, TextGenerationOptions, TextGenerationOptionsBuilder};
use tracing::debug;
use utoipa::ToSchema;
use uuid::Uuid;

use super::model;
use crate::{fatal, Device};

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
#[schema(example=json!({
    "messages": [
        Message { role: "user".to_owned(), content: "What is tail recursion?".to_owned()},
        Message { role: "assistant".to_owned(), content: "It's a kind of optimization in compiler?".to_owned()},
        Message { role: "user".to_owned(), content: "Could you share more details?".to_owned()},
    ]
}))]
pub struct ChatCompletionRequest {
    messages: Vec<Message>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct ChatCompletionChunk {
    id: String,
    created: u64,
    system_fingerprint: String,
    object: &'static str,
    model: &'static str,
    choices: [ChatCompletionChoice; 1],
}

#[derive(Serialize, Deserialize, Clone, Debug, ToSchema)]
pub struct ChatCompletionChoice {
    index: usize,
    logprobs: Option<String>,
    finish_reason: Option<String>,
    delta: ChatCompletionDelta,
}

#[derive(Serialize, Deserialize, Clone, Debug, ToSchema)]
pub struct ChatCompletionDelta {
    content: String,
}

impl ChatCompletionChunk {
    fn new(content: String, id: String, created: u64, last_chunk: bool) -> Self {
        ChatCompletionChunk {
            id,
            created,
            object: "chat.completion.chunk",
            model: "unused-model",
            system_fingerprint: "unused-system-fingerprint".into(),
            choices: [ChatCompletionChoice {
                index: 0,
                delta: ChatCompletionDelta { content },
                logprobs: None,
                finish_reason: last_chunk.then(|| "stop".into()),
            }],
        }
    }
}

pub struct ChatService {
    engine: Arc<dyn TextGeneration>,
    prompt_builder: ChatPromptBuilder,
}

impl ChatService {
    fn new(engine: Arc<dyn TextGeneration>, chat_template: String) -> Self {
        Self {
            engine,
            prompt_builder: ChatPromptBuilder::new(chat_template),
        }
    }

    fn text_generation_options() -> TextGenerationOptions {
        TextGenerationOptionsBuilder::default()
            .max_input_length(2048)
            .max_decoding_length(1920)
            .sampling_temperature(0.1)
            .build()
            .unwrap()
    }

    pub async fn generate(
        &self,
        request: &ChatCompletionRequest,
    ) -> BoxStream<ChatCompletionChunk> {
        let prompt = self.prompt_builder.build(&request.messages);
        let options = Self::text_generation_options();
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Must be able to read system clock")
            .as_secs();
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        debug!("PROMPT: {}", prompt);
        let s = stream! {
            for await content in self.engine.generate_stream(&prompt, options).await {
                yield ChatCompletionChunk::new(content, id.clone(), created, false)
            }
            yield ChatCompletionChunk::new("".into(), id, created, true)
        };

        Box::pin(s)
    }
}

pub async fn create_chat_service(model: &str, device: &Device, parallelism: u8) -> ChatService {
    let (engine, model::PromptInfo { chat_template, .. }) =
        model::load_text_generation(model, device, parallelism).await;

    let Some(chat_template) = chat_template else {
        fatal!("Chat model requires specifying prompt template");
    };

    ChatService::new(engine, chat_template)
}
