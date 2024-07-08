use anyhow::{Context, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{ChatCompletionRequestMessage, CreateChatCompletionRequestArgs},
};
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use tabby_common::api::chat::Message;
use tabby_inference::{ChatCompletionOptions, ChatCompletionStream};
use tracing::{debug, warn};

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
impl ChatCompletionStream for OpenAIChatEngine {
    async fn chat_completion(
        &self,
        input_messages: &[Message],
        options: ChatCompletionOptions,
    ) -> Result<BoxStream<String>> {
        let messages = input_messages.to_vec();
        let request = CreateChatCompletionRequestArgs::default()
            .seed(options.seed as i64)
            .max_tokens(options.max_decoding_tokens as u16)
            .model(&self.model_name)
            .temperature(options.sampling_temperature)
            .presence_penalty(options.presence_penalty)
            .stream(true)
            .messages(
                serde_json::from_value::<Vec<ChatCompletionRequestMessage>>(serde_json::to_value(
                    messages,
                )?)
                .context("Failed to parse from json")?,
            )
            .build()?;

        debug!("openai-chat request: {:?}", request);
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
                        yield x.choices[0].delta.content.clone().unwrap_or_default();
                    },
                    Err(e) => {
                        // Stream finished.
                        debug!("openai-chat stream finished: {:?}", e);
                        break;
                    }
                };
            }
        };

        Ok(Box::pin(s))
    }
}
