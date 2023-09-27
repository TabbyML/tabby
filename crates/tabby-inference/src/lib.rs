use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use futures::stream::BoxStream;

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
    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String> {
        let prompt = prompt.to_owned();
        let stream = stream! {
            yield self.generate(&prompt, options).await;
        };

        Box::pin(stream)
    }
}

#[cfg(test)]
mod tests {

    use futures::{pin_mut, StreamExt};

    use super::{async_trait, TextGeneration, TextGenerationOptions, TextGenerationOptionsBuilder};

    #[derive(Default)]
    struct FakeEngine {}

    #[async_trait]
    impl TextGeneration for FakeEngine {
        async fn generate(&self, _prompt: &str, _options: TextGenerationOptions) -> String {
            "fake".to_owned()
        }
    }

    #[tokio::test]
    async fn it_generate() {
        let engine = FakeEngine::default();
        let options = TextGenerationOptionsBuilder::default().build().unwrap();
        let ret = engine.generate("", options).await;
        assert_eq!(ret, "fake");
    }

    #[tokio::test]
    async fn it_generate_stream() {
        let engine = FakeEngine::default();
        let options = TextGenerationOptionsBuilder::default().build().unwrap();
        let stream = engine.generate_stream("", options).await;

        pin_mut!(stream);

        let ret = stream.next().await.unwrap();
        assert_eq!(ret, "fake");

        let ret = stream.next().await;
        assert!(ret.is_none());
    }
}
