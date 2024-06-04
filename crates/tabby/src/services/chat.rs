use std::sync::Arc;

use async_stream::stream;
use derive_builder::Builder;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use tabby_common::{
    api::{
        chat::Message,
        event::{Event, EventLogger},
    },
    config::ModelConfig,
};
use tabby_inference::{ChatCompletionOptionsBuilder, ChatCompletionStream};
use tracing::warn;
use utoipa::ToSchema;
use uuid::Uuid;

use super::model;

#[derive(Serialize, Deserialize, ToSchema, Clone, Builder, Debug)]
#[schema(example=json!({
    "messages": [
        Message { role: "user".to_owned(), content: "What is tail recursion?".to_owned()},
        Message { role: "assistant".to_owned(), content: "It's a kind of optimization in compiler?".to_owned()},
        Message { role: "user".to_owned(), content: "Could you share more details?".to_owned()},
    ]
}))]
pub struct ChatCompletionRequest {
    #[builder(default = "None")]
    pub(crate) user: Option<String>,

    messages: Vec<Message>,

    #[builder(default = "None")]
    temperature: Option<f32>,

    #[builder(default = "None")]
    seed: Option<u64>,

    #[builder(default = "None")]
    presence_penalty: Option<f32>,
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
            request.presence_penalty.inspect(|x| {
                builder.presence_penalty(*x);
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

            self.logger.log(request.user, Event::ChatCompletion {
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

pub async fn create_chat_service(logger: Arc<dyn EventLogger>, chat: &ModelConfig) -> ChatService {
    let engine = model::load_chat_completion(chat).await;

    ChatService::new(engine, logger)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use anyhow::Result;
    use async_trait::async_trait;
    use futures::StreamExt;
    use tabby_inference::ChatCompletionOptions;

    use super::*;

    struct MockChatCompletionStream;

    #[async_trait]
    impl ChatCompletionStream for MockChatCompletionStream {
        async fn chat_completion(
            &self,
            _messages: &[Message],
            _options: ChatCompletionOptions,
        ) -> Result<BoxStream<String>> {
            let s = stream! {
                yield "Hello, world!".into();
            };
            Ok(Box::pin(s))
        }
    }

    struct MockEventLogger(Mutex<Vec<Event>>);

    impl EventLogger for MockEventLogger {
        fn write(&self, x: tabby_common::api::event::LogEntry) {
            self.0.lock().unwrap().push(x.event);
        }
    }

    #[tokio::test]
    async fn test_chat_service() {
        let engine = Arc::new(MockChatCompletionStream);
        let logger = Arc::new(MockEventLogger(Default::default()));
        let service = Arc::new(ChatService::new(engine, logger.clone()));

        let request = ChatCompletionRequest {
            messages: vec![Message {
                role: "user".into(),
                content: "Hello, computer!".into(),
            }],
            temperature: None,
            seed: None,
            presence_penalty: None,
            user: None,
        };
        let mut output = service.generate(request).await;
        let response = output.next().await.unwrap();
        assert_eq!(response.choices[0].delta.content, "Hello, world!");

        let finish = output.next().await.unwrap();
        assert_eq!(finish.choices[0].delta.content, "");
        assert_eq!(finish.choices[0].finish_reason.as_ref().unwrap(), "stop");

        assert!(output.next().await.is_none());

        let event = &logger.0.lock().unwrap()[0];
        let Event::ChatCompletion { output, .. } = event else {
            panic!("Expected ChatCompletion event");
        };
        assert_eq!(output.role, "assistant");
        assert_eq!(output.content, "Hello, world!");
    }
}
