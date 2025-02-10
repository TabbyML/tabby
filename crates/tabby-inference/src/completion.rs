use async_trait::async_trait;
use derive_builder::Builder;
use futures::stream::BoxStream;

#[derive(Builder, Debug)]
pub struct CompletionOptions {
    pub max_decoding_tokens: i32,

    pub sampling_temperature: f32,

    pub seed: u64,

    #[builder(default = "0.0")]
    pub presence_penalty: f32,
}

#[async_trait]
pub trait CompletionStream: Sync + Send {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String>;
}
