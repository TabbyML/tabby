use std::sync::Arc;

use async_stream::stream;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use tabby_common::api::{
    chat::Message,
    event::{Event, EventLogger},
};
use tabby_inference::{ChatCompletionOptionsBuilder, ChatCompletionStream};
use tracing::warn;
use utoipa::ToSchema;
use uuid::Uuid;
use derive_builder::Builder;

use super::model;
use crate::Device;

#[derive(Serialize, Deserialize, ToSchema, Clone, Builder, Debug)]
#[schema(example=json!({
    "messages": [
        Message { role: "user".to_owned(), content: "What is tail recursion?".to_owned()},
        Message { role: "assistant".to_owned(), content: "It's a kind of optimization in compiler?".to_owned()},
        Message { role: "user".to_owned(), content: "Could you share more details?".to_owned()},
    ]
}))]
pub struct ChatCompletionRequest {
    messages: Vec<Message>,

    #[builder(default = "None")]
    temperature: Option<f32>,

    #[builder(default = "None")]
    seed: Option<u64>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct ChatCompletionChunk {
    id: String,
    created: u64,
    system_fingerprint: String,
    object: &'static str,
    model: &'static str,
    pub choices: [ChatCompletionChoice; 1],
}

#[derive(Serialize, Deserialize, Clone, Debug, ToSchema)]
pub struct ChatCompletionChoice {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    pub delta: ChatCompletionDelta,
}

#[derive(Serialize, Deserialize, Clone, Debug, ToSchema)]
pub struct ChatCompletionDelta {
    pub content: String,
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
    engine: Arc<dyn ChatCompletionStream>,
    logger: Arc<dyn EventLogger>,
}

impl ChatService {
    fn new(engine: Arc<dyn ChatCompletionStream>, logger: Arc<dyn EventLogger>) -> Self {
        Self { engine, logger }
    }

    pub async fn generate<'a>(
        self: Arc<Self>,
        request: ChatCompletionRequest,
    ) -> BoxStream<'a, ChatCompletionChunk> {
        let mut output = String::new();

        let options = {
            let mut builder = ChatCompletionOptionsBuilder::default();
            request.temperature.inspect(|x| {
                builder.sampling_temperature(*x);
            });
            request.seed.inspect(|x| {
                builder.seed(*x);
            });
            builder
                .build()
                .expect("Failed to create ChatCompletionOptions")
        };

        let s = stream! {
            let s = match self.engine.chat_completion(&request.messages, options).await {
                Ok(x) => x,
                Err(e) => {
                    warn!("Failed to start chat completion: {:?}", e);
                    return;
                }
            };

            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Must be able to read system clock")
                .as_secs();

            let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
            for await content in s {
                output.push_str(&content);
                yield ChatCompletionChunk::new(content, completion_id.clone(), created, false);
            }
            yield ChatCompletionChunk::new(String::default(), completion_id.clone(), created, true);

            // FIXME(boxbeam): Log user for chat completion events
            self.logger.log(None, Event::ChatCompletion {
                completion_id,
                input: convert_messages(&request.messages),
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
    let engine = model::load_chat_completion(model, device, parallelism).await;

    ChatService::new(engine, logger)
}
