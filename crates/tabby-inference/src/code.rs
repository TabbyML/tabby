use std::sync::Arc;

use async_stream::stream;
use derive_builder::Builder;
use futures::StreamExt;
use tabby_common::{config::ModelConfig, languages::Language};

use crate::{decoding::StopConditionFactory, CompletionOptionsBuilder, CompletionStream};

#[derive(Builder, Debug)]
pub struct CodeGenerationOptions {
    #[builder(default = "1024")]
    pub max_input_length: usize,

    #[builder(default = "256")]
    pub max_decoding_tokens: i32,

    #[builder(default = "0.1")]
    pub sampling_temperature: f32,

    #[builder(default = "crate::default_seed()")]
    pub seed: u64,

    #[builder(default = "None")]
    pub language: Option<&'static Language>,
}

pub struct CodeGeneration {
    imp: Arc<dyn CompletionStream>,
    stop_condition_factory: StopConditionFactory,
}

impl CodeGeneration {
    pub fn new(imp: Arc<dyn CompletionStream>, config: Option<ModelConfig>) -> Self {
        let additional_stop_words = match config {
            Some(ModelConfig::Local(config)) => config.additional_stop_words.unwrap_or_default(),
            Some(ModelConfig::Http(config)) => config.additional_stop_words.unwrap_or_default(),
            _ => vec![],
        };
        let stop_condition_factory = StopConditionFactory::with_stop_words(additional_stop_words);

        Self {
            imp,
            stop_condition_factory,
        }
    }
}

impl CodeGeneration {
    pub async fn generate(&self, prompt: &str, options: CodeGenerationOptions) -> String {
        let s = stream! {
            let mut text = String::new();
            let mut stop_condition = self.stop_condition_factory.create(
                prompt,
                options.language,
            );

            let options = CompletionOptionsBuilder::default()
                .max_input_length(options.max_input_length)
                .max_decoding_tokens(options.max_decoding_tokens)
                .sampling_temperature(options.sampling_temperature)
                .seed(options.seed)
                .build()
                .expect("Failed to build completion options");

            for await new_text in self.imp.generate(prompt, options).await {
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

        Box::pin(s).into_future().await.0.unwrap_or_default()
    }
}
