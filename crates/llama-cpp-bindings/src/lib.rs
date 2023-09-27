use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use ffi::create_engine;
use futures::{
    lock::Mutex,
    pin_mut,
    stream::{BoxStream, StreamExt},
};
use stop_words::StopWords;
use tabby_inference::{TextGeneration, TextGenerationOptions};
use tokenizers::tokenizer::Tokenizer;

#[cxx::bridge(namespace = "llama")]
mod ffi {
    unsafe extern "C++" {
        include!("llama-cpp-bindings/include/engine.h");

        type TextInferenceEngine;

        fn create_engine(model_path: &str) -> SharedPtr<TextInferenceEngine>;

        fn start(&self, prompt: &str, max_input_length: usize) -> u32;
        fn step(&self, next_token_id: u32) -> u32;
        fn end(&self);

        fn eos_token(&self) -> u32;
    }
}

unsafe impl Send for ffi::TextInferenceEngine {}
unsafe impl Sync for ffi::TextInferenceEngine {}

#[derive(Builder, Debug)]
pub struct LlamaEngineOptions {
    model_path: String,
    tokenizer_path: String,
}

pub struct LlamaEngine {
    engine: Mutex<cxx::SharedPtr<ffi::TextInferenceEngine>>,
    tokenizer: Arc<Tokenizer>,
    stop_words: StopWords,
}

impl LlamaEngine {
    pub fn create(options: LlamaEngineOptions) -> Self {
        LlamaEngine {
            engine: Mutex::new(create_engine(&options.model_path)),
            tokenizer: Arc::new(Tokenizer::from_file(&options.tokenizer_path).unwrap()),
            stop_words: StopWords::default(),
        }
    }
}

#[async_trait]
impl TextGeneration for LlamaEngine {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let s = self.generate_stream(prompt, options).await;
        pin_mut!(s);

        let mut text = "".to_owned();
        while let Some(value) = s.next().await {
            text += &value;
        }

        text
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String> {
        let prompt = prompt.to_owned();
        let mut stop_condition = self
            .stop_words
            .create_condition(self.tokenizer.clone(), options.stop_words);

        let s = stream! {
            let engine = self.engine.lock().await;
            let eos_token = engine.eos_token();

            let mut next_token_id = engine.start(&prompt, options.max_input_length);
            if next_token_id == eos_token {
                yield "".to_owned();
            } else {
                let mut n_remains = options.max_decoding_length - 1;

                while n_remains > 0 {
                    next_token_id = engine.step(next_token_id);
                    if next_token_id == eos_token {
                        break;
                    }

                    if stop_condition.next_token(next_token_id) {
                        break;
                    }

                    let text = self.tokenizer.decode(&[next_token_id], true).unwrap();
                    yield text;
                    n_remains -= 1;
                }
            }

            engine.end();
        };

        Box::pin(s)
    }
}
