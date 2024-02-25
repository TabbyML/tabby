mod chat_prompt;

use std::sync::Arc;

use async_stream::stream;
use chat_prompt::ChatPromptBuilder;
use futures::stream::BoxStream;
use tabby_common::api::{chat::{ChatCompletionChunk, ChatCompletionRequest, Message}, event::{Event, EventLogger}};
use tabby_inference::{
    chat::{self, ChatCompletionStreaming},
    TextGeneration, TextGenerationOptions, TextGenerationStream,
};
use tracing::warn;

use super::model;
use crate::{fatal, Device};

pub struct ChatService {
    engine: Arc<dyn TextGeneration>,
    logger: Arc<dyn EventLogger>,
    prompt_builder: ChatPromptBuilder,
}

impl chat::ChatPromptBuilder for ChatService {
    fn build_chat_prompt(&self, messages: &[Message]) -> anyhow::Result<String> {
        self.prompt_builder.build(messages)
    }
}

#[async_trait::async_trait]
impl TextGenerationStream for ChatService {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> BoxStream<String> {
        let prompt = prompt.to_owned();
        let s = stream! {
            for await (streaming, text) in self.engine.generate_stream(&prompt, options).await {
                if !streaming {
                    yield text;
                }
            }
        };

        Box::pin(s)
    }
}

impl ChatService {
    fn new(
        engine: Arc<dyn TextGeneration>,
        logger: Arc<dyn EventLogger>,
        chat_template: String,
    ) -> Self {
        Self {
            engine,
            logger,
            prompt_builder: ChatPromptBuilder::new(chat_template),
        }
    }

    pub async fn generate<'a>(
        self: Arc<Self>,
        request: ChatCompletionRequest,
    ) -> BoxStream<'a, ChatCompletionChunk> {
        let mut output = String::new();
        let input = convert_messages(&request.messages);
        let s = stream! {
            let s = match self.chat_completion(request).await {
                Ok(x) => x,
                Err(e) => {
                    warn!("Failed to start chat completion: {:?}", e);
                    return;
                }
            };

            let mut completion_id = String::default();
            for await chunk in s {
                if completion_id.is_empty() {
                    completion_id = chunk.id.clone();
                }
                output.push_str(&chunk.choices[0].delta.content);
                yield chunk
            }

            self.logger.log(Event::ChatCompletion {
                completion_id,
                input,
                output: create_assistant_message(output)
            });
        };

        Box::pin(s)
    }
}

fn create_assistant_message(string: String) -> tabby_common::api::event::Message {
    tabby_common::api::event::Message {
        content: string,
        role: "assistant".into(),
    }
}

fn convert_messages(input: &[Message]) -> Vec<tabby_common::api::event::Message> {
    input
        .iter()
        .map(|m| tabby_common::api::event::Message {
            content: m.content.clone(),
            role: m.role.clone(),
        })
        .collect()
}

pub async fn create_chat_service(
    logger: Arc<dyn EventLogger>,
    model: &str,
    device: &Device,
    parallelism: u8,
) -> ChatService {
    let (engine, model::PromptInfo { chat_template, .. }) =
        model::load_text_generation(model, device, parallelism).await;

    let Some(chat_template) = chat_template else {
        fatal!("Chat model requires specifying prompt template");
    };

    ChatService::new(engine, logger, chat_template)
}
