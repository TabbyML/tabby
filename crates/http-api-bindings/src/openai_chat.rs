use anyhow::{Context, Result};
use async_openai::{config::OpenAIConfig, types::CreateChatCompletionRequest};
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use tabby_common::api::chat::{ChatCompletionChunk, ChatCompletionRequest};
use tabby_inference::chat::ChatCompletionStreaming;
use tracing::warn;

pub struct OpenAIChatEngine {
    client: async_openai::Client<OpenAIConfig>,
    model_name: String,
}

impl OpenAIChatEngine {
    pub fn create(api_endpoint: &str, model_name: &str, api_key: Option<String>) -> Self {
        let config = OpenAIConfig::default()
            .with_api_base(api_endpoint)
            .with_api_key(api_key.unwrap_or_default());

        let client = async_openai::Client::with_config(config);

        Self {
            client,
            model_name: model_name.to_owned(),
        }
    }
}

#[async_trait]
impl ChatCompletionStreaming for OpenAIChatEngine {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<BoxStream<ChatCompletionChunk>> {
        let mut request = request;
        request.model = Some(self.model_name.clone());

        let mut request: CreateChatCompletionRequest =
            serde_json::from_value(serde_json::to_value(request)?)
                .context("Failed to parse from json")?;
        request.stream = Some(true);

        let s = stream! {
            let s = match self.client.chat().create_stream(request).await {
                Ok(x) => x,
                Err(e) => {
                    warn!("Failed to create completion request {:?}", e);
                    return;
                }
            };

            for await x in s {
                match x {
                    Ok(x) => {
                        let choice = x.choices[0].clone();
                        yield ChatCompletionChunk::new(
                            choice.delta.content.unwrap_or_default(),
                            x.id,
                            x.created as u64,
                            choice.finish_reason.is_some()
                        )
                    },
                    Err(e) => {
                        warn!("Failed to stream response: {}", e);
                        break;
                    }
                };
            }
        };

        Ok(Box::pin(s))
    }
}
