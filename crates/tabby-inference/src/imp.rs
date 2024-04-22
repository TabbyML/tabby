use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};

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

    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<(bool, String)> {
        let prompt = prompt.to_owned();
        let s = stream! {
            let mut stop_condition = self.stop_condition_factory.create(
                &prompt,
                options.max_decoding_length,
                options.language,
            );

            let mut text = String::new();
            for await new_text in self.imp.generate(&prompt, options).await {
                let (should_stop, stop_length) = stop_condition.should_stop(&new_text);
                text += &new_text;
                yield (true, new_text);
                if should_stop {
                    // stop condition matched against prompt + generated text. There's a chance that stop_length >= text.len();
                    let new_text_length = text.len().checked_sub(stop_length).unwrap_or_default();
                    text.truncate(new_text_length);
                    yield (false, text);
                    return
                }
            }

            yield (false, text);
        };
        Box::pin(s)
    }
}

#[async_trait]
impl<T: TextGenerationStream> TextGeneration for TextGenerationImpl<T> {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let prompt = prompt.to_owned();
        let s = stream! {
            for await (streaming, text) in self.generate_stream(&prompt, options).await {
                if !streaming {
                    yield text;
                }
            }
        };

        if let Some(text) = Box::pin(s).into_future().await.0 {
            text
        } else {
            String::new()
        }
    }
}
