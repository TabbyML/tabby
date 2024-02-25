use anyhow::Result;
use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use futures::stream::BoxStream;
use tabby_common::api::chat::Message;

use crate::{TextGenerationOptions, TextGenerationOptionsBuilder, TextGenerationStream};

#[derive(Builder, Debug)]
pub struct ChatCompletionOptions {
    #[builder(default = "0.1")]
    pub sampling_temperature: f32,

    #[builder(default = "TextGenerationOptions::default_seed()")]
    pub seed: u64,
}

#[async_trait]
pub trait ChatCompletionStream: Sync + Send {
    async fn chat_completion(
        &self,
        messages: &[Message],
        options: ChatCompletionOptions,
    ) -> Result<BoxStream<String>>;
}

pub trait ChatPromptBuilder {
    fn build_chat_prompt(&self, messages: &[Message]) -> Result<String>;
}

#[async_trait]
impl<T: ChatPromptBuilder + TextGenerationStream> ChatCompletionStream for T {
    async fn chat_completion(
        &self,
        messages: &[Message],
        options: ChatCompletionOptions,
    ) -> Result<BoxStream<String>> {
        let options = TextGenerationOptionsBuilder::default()
            .max_input_length(2048)
            .max_decoding_length(1920)
            .seed(options.seed)
            .sampling_temperature(options.sampling_temperature)
            .build()?;

        let prompt = self.build_chat_prompt(messages)?;

        let s = stream! {
            for await content in self.generate(&prompt, options).await {
                yield content
            }
        };

        Ok(Box::pin(s))
    }
}
