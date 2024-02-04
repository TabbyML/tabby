use async_stream::stream;
use async_trait::async_trait;
use futures::{pin_mut, stream::BoxStream, Stream, StreamExt};

use crate::{
    decoding::StopConditionFactory, TextGeneration, TextGenerationOptions, TextGenerationStream,
};

pub struct TextGenerationImpl<T: TextGenerationStream> {
    imp: T,
    stop_condition_factory: StopConditionFactory,
}

impl<T: TextGenerationStream> TextGenerationImpl<T> {
    pub fn new(imp: T) -> Self {
        Self {
            imp,
            stop_condition_factory: StopConditionFactory::default(),
        }
    }
}

#[async_trait]
impl<T: TextGenerationStream> TextGeneration for TextGenerationImpl<T> {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let language = options.language;
        let s = self.generate_stream(prompt, options).await;
        let text = stream_to_string(s).await;

        let Some(language) = language else {
            return text;
        };

        let Some(trimmed) = self.stop_condition_factory.trim_stop_words(language, &text) else {
            return text;
        };

        trimmed
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String> {
        let prompt = prompt.to_owned();
        let s = stream! {
            let mut stop_condition = self.stop_condition_factory.create(
                &prompt,
                options.max_decoding_length,
                options.language,
            );

            for await new_text in self.imp.generate(&prompt, options).await {
                let should_stop = stop_condition.should_stop(&new_text);
                yield new_text;
                if should_stop {
                    break;
                }
            }
        };
        Box::pin(s)
    }
}

async fn stream_to_string(s: impl Stream<Item = String>) -> String {
    pin_mut!(s);

    let mut text = "".to_owned();
    while let Some(value) = s.next().await {
        text += &value;
    }

    text
}
