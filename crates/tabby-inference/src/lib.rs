use async_trait::async_trait;
use derive_builder::Builder;

#[derive(Builder, Debug)]
pub struct TextGenerationOptions {
    #[builder(default = "256")]
    pub max_decoding_length: usize,

    #[builder(default = "1.0")]
    pub sampling_temperature: f32,

    pub stop_words: &'static Vec<&'static str>,
}

#[async_trait]
pub trait TextGeneration : Sync + Send {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String;
}
