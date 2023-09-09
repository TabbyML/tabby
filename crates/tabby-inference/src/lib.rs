use async_trait::async_trait;
use derive_builder::Builder;

#[derive(Builder, Debug)]
pub struct TextGenerationOptions {
    #[builder(default = "1024")]
    pub max_input_length: usize,

    #[builder(default = "256")]
    pub max_decoding_length: usize,

    #[builder(default = "1.0")]
    pub sampling_temperature: f32,

    #[builder(default = "&EMPTY_STOP_WORDS")]
    pub stop_words: &'static Vec<&'static str>,
}

static EMPTY_STOP_WORDS: Vec<&'static str> = vec![];

#[async_trait]
pub trait TextGeneration: Sync + Send {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String;
}
