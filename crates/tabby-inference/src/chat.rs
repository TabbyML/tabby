use anyhow::Result;
use async_trait::async_trait;
use derive_builder::Builder;
use futures::stream::BoxStream;
use tabby_common::api::chat::Message;

#[derive(Builder, Debug)]
pub struct ChatCompletionOptions {
    #[builder(default = "0.1")]
    pub sampling_temperature: f32,

    #[builder(default = "crate::default_seed()")]
    pub seed: u64,

    #[builder(default = "1920")]
    pub max_decoding_tokens: i32,
}

#[async_trait]
pub trait ChatCompletionStream: Sync + Send {
    async fn chat_completion(
        &self,
        messages: &[Message],
        options: ChatCompletionOptions,
    ) -> Result<BoxStream<String>>;
}
