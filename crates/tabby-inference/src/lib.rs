pub mod decoding;

use async_trait::async_trait;
use derive_builder::Builder;
use futures::stream::BoxStream;
use tabby_common::languages::Language;

#[derive(Builder, Debug)]
pub struct TextGenerationOptions {
    #[builder(default = "1024")]
    pub max_input_length: usize,

    #[builder(default = "256")]
    pub max_decoding_length: usize,

    #[builder(default = "1.0")]
    pub sampling_temperature: f32,

    #[builder(default = "None")]
    pub language: Option<&'static Language>,
}

#[async_trait]
pub trait TextGeneration: Sync + Send {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String;
    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String>;
}

pub mod helpers {
    use async_stream::stream;
    use futures::{pin_mut, stream::BoxStream, Stream, StreamExt};

    pub async fn stream_to_string(s: impl Stream<Item = String>) -> String {
        pin_mut!(s);

        let mut text = "".to_owned();
        while let Some(value) = s.next().await {
            text += &value;
        }

        text
    }

    pub async fn string_to_stream(s: String) -> BoxStream<'static, String> {
        let stream = stream! {
            yield s
        };

        Box::pin(stream)
    }
}
