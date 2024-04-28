use anyhow::Result;
use async_trait::async_trait;
use derive_builder::Builder;
use futures::stream::BoxStream;
use tabby_common::api::chat::Message;

use crate::TextGenerationOptions;

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
