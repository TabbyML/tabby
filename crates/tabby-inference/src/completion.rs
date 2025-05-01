use async_trait::async_trait;
use derive_builder::Builder;
use futures::{stream::BoxStream, StreamExt};

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
    /// Generate a completion in streaming mode
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String>;

    /// Generate a completion in non-streaming mode
    /// Returns the full completion as a single string
    async fn generate_sync(&self, prompt: &str, options: CompletionOptions) -> String {
        let mut stream = self.generate(prompt, options).await;
        let mut result = String::new();
        while let Some(chunk) = stream.next().await {
            result.push_str(&chunk);
        }
        result
    }
}
