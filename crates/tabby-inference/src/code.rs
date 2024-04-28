use std::sync::Arc;

use async_stream::stream;
use derive_builder::Builder;
use futures::StreamExt;
use tabby_common::languages::Language;

use crate::{decoding::StopConditionFactory, CompletionOptions, CompletionStream};

#[derive(Builder, Debug)]
pub struct CodeGenerationOptions {
    #[builder(default = "1024")]
    pub max_input_length: usize,

    #[builder(default = "256")]
    pub max_decoding_length: usize,

    #[builder(default = "0.1")]
    pub sampling_temperature: f32,

    #[builder(default = "crate::default_seed()")]
    pub seed: u64,

    #[builder(default = "None")]
    pub language: Option<&'static Language>,
}

impl From<CodeGenerationOptions> for CompletionOptions {
    fn from(val: CodeGenerationOptions) -> Self {
        CompletionOptions {
            max_input_length: val.max_input_length,
            sampling_temperature: val.sampling_temperature,
            seed: val.seed,
        }
    }
}

pub struct CodeGeneration {
    imp: Arc<dyn CompletionStream>,
    stop_condition_factory: StopConditionFactory,
}

impl CodeGeneration {
    pub fn new(imp: Arc<dyn CompletionStream>) -> Self {
        Self {
            imp,
            stop_condition_factory: StopConditionFactory::default(),
        }
    }
}

impl CodeGeneration {
    pub async fn generate(&self, prompt: &str, options: CodeGenerationOptions) -> String {
        let s = stream! {
            let mut text = String::new();
            let mut stop_condition = self.stop_condition_factory.create(
                prompt,
                options.max_decoding_length,
                options.language,
            );

            for await new_text in self.imp.generate(prompt, options.into()).await {
                let (should_stop, stop_length) = stop_condition.should_stop(&new_text);
                text += &new_text;
                if should_stop {
                    // stop condition matched against prompt + generated text. There's a chance that stop_length >= text.len();
                    let new_text_length = text.len().checked_sub(stop_length).unwrap_or_default();
                    text.truncate(new_text_length);
                    break;
                }
            }

            yield text;
        };

        if let Some(text) = Box::pin(s).into_future().await.0 {
            text
        } else {
            String::new()
        }
    }
}
