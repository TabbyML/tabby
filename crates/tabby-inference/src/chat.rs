use anyhow::Result;
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use tabby_common::api::chat as types;
use uuid::Uuid;

use crate::{TextGenerationOptions, TextGenerationOptionsBuilder, TextGenerationStream};

#[async_trait]
pub trait ChatCompletionStreaming: Sync + Send {
    async fn chat_completion(
        &self,
        request: types::ChatCompletionRequest,
    ) -> Result<BoxStream<types::ChatCompletionChunk>>;
}

pub trait ChatPromptBuilder {
    fn build_chat_prompt(&self, messages: &[types::Message]) -> Result<String>;
}

#[async_trait]
impl<T: ChatPromptBuilder + TextGenerationStream> ChatCompletionStreaming for T {
    async fn chat_completion(
        &self,
        request: types::ChatCompletionRequest,
    ) -> Result<BoxStream<types::ChatCompletionChunk>> {
        let options = TextGenerationOptionsBuilder::default()
            .max_input_length(2048)
            .max_decoding_length(1920)
            .seed(request.seed.unwrap_or_else(TextGenerationOptions::default_seed))
            .sampling_temperature(request.temperature.unwrap_or(0.1))
            .build()?;

        let prompt = self.build_chat_prompt(&request.messages)?;

        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Must be able to read system clock")
            .as_secs();
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let s = stream! {
            for await content in self.generate(&prompt, options).await {
                yield types::ChatCompletionChunk::new(content, id.clone(), created, false);
            }

            yield types::ChatCompletionChunk::new(String::default(), id.clone(), created, true);
        };

        Ok(Box::pin(s))
    }
}
