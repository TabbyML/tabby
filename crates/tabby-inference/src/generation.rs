use async_stream::stream;
use futures::stream::BoxStream;

use crate::{decoding::StopConditionFactory, TextGenerationOptions, TextGenerationStream};

pub struct TextGeneration {
    imp: Box<dyn TextGenerationStream>,
    stop_condition_factory: StopConditionFactory,
}

impl TextGeneration {
    pub fn new(imp: impl TextGenerationStream + 'static) -> Self {
        Self {
            imp: Box::new(imp),
            stop_condition_factory: StopConditionFactory::default(),
        }
    }
}

impl TextGeneration {
    pub async fn generate_stream(
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
