use std::sync::Arc;

use async_stream::stream;
use futures::stream::BoxStream;
use tabby_common::api::{
    chat::{ChatCompletionChunk, ChatCompletionRequest, Message},
    event::{Event, EventLogger},
};
use tabby_inference::chat::ChatCompletionStreaming;
use tracing::warn;

use super::model;
use crate::Device;

pub struct ChatService {
    engine: Arc<dyn ChatCompletionStreaming>,
    logger: Arc<dyn EventLogger>,
}

impl ChatService {
    fn new(engine: Arc<dyn ChatCompletionStreaming>, logger: Arc<dyn EventLogger>) -> Self {
        Self { engine, logger }
    }

    pub async fn generate<'a>(
        self: Arc<Self>,
        request: ChatCompletionRequest,
    ) -> BoxStream<'a, ChatCompletionChunk> {
        let mut output = String::new();
        let input = convert_messages(&request.messages);
        let s = stream! {
            let s = match self.engine.chat_completion(request).await {
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
    let engine = model::load_chat_completion(model, device, parallelism).await;

    ChatService::new(engine, logger)
}
