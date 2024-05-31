use std::sync::Arc;

use anyhow::{bail, Result};
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use ollama_rs::{
    generation::{
        chat::{request::ChatMessageRequest, ChatMessage, MessageRole},
        options::GenerationOptions,
    },
    Ollama,
};
use tabby_common::{api::chat::Message, config::HttpModelConfig};
use tabby_inference::{ChatCompletionOptions, ChatCompletionStream};

use crate::model::OllamaModelExt;

/// A special adapter to convert Tabby messages to ollama-rs messages
struct ChatMessageAdapter(ChatMessage);

impl TryFrom<Message> for ChatMessageAdapter {
    type Error = anyhow::Error;
    fn try_from(value: Message) -> Result<ChatMessageAdapter> {
        let role = match value.role.as_str() {
            "system" => MessageRole::System,
            "assistant" => MessageRole::Assistant,
            "user" => MessageRole::User,
            other => bail!("Unsupported chat message role: {other}"),
        };

        Ok(ChatMessageAdapter(ChatMessage::new(role, value.content)))
    }
}

impl From<ChatMessageAdapter> for ChatMessage {
    fn from(val: ChatMessageAdapter) -> Self {
        val.0
    }
}

/// Ollama chat completions
pub struct OllamaChat {
    /// Connection to Ollama API
    connection: Ollama,
    /// Model name, <model>
    model: String,
}

#[async_trait]
impl ChatCompletionStream for OllamaChat {
    async fn chat_completion(
        &self,
        messages: &[Message],
        options: ChatCompletionOptions,
    ) -> Result<BoxStream<String>> {
        let messages = messages
            .iter()
            .map(|m| ChatMessageAdapter::try_from(m.to_owned()))
            .collect::<Result<Vec<_>, _>>()?;

        let messages = messages.into_iter().map(|m| m.into()).collect::<Vec<_>>();

        let options = GenerationOptions::default()
            .seed(options.seed as i32)
            .temperature(options.sampling_temperature)
            .num_predict(options.max_decoding_tokens);

        let request = ChatMessageRequest::new(self.model.to_owned(), messages).options(options);

        let stream = self.connection.send_chat_messages_stream(request).await?;

        let stream = stream
            .map(|x| match x {
                Ok(response) => response.message,
                Err(_) => None,
            })
            .map(|x| match x {
                Some(e) => e.content,
                None => "".to_owned(),
            });

        Ok(stream.boxed())
    }
}

pub async fn create(config: &HttpModelConfig) -> Arc<dyn ChatCompletionStream> {
    let connection = Ollama::try_new(config.api_endpoint.to_owned())
        .expect("Failed to create connection to Ollama, URL invalid");

    let model = connection.select_model_or_default(config).await.unwrap();

    Arc::new(OllamaChat { connection, model })
}
