use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use derive_builder::Builder;
use ffi::create_engine;
use stop_words::StopWords;
use tabby_inference::{TextGeneration, TextGenerationOptions};
use tokenizers::tokenizer::Tokenizer;
use tokio_util::sync::CancellationToken;

#[cxx::bridge(namespace = "llama")]
mod ffi {
    unsafe extern "C++" {
        include!("llama-cpp-bindings/include/engine.h");

        type TextInferenceEngine;

        fn create_engine(model_path: &str) -> SharedPtr<TextInferenceEngine>;

        fn start(&self, prompt: &str) -> u32;
        fn step(&self, next_token_id: u32) -> u32;
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
    engine: Arc<Mutex<cxx::SharedPtr<ffi::TextInferenceEngine>>>,
    tokenizer: Arc<Tokenizer>,
    stop_words: StopWords,
}

impl LlamaEngine {
    pub fn create(options: LlamaEngineOptions) -> Self {
        LlamaEngine {
            engine: Arc::new(Mutex::new(create_engine(&options.model_path))),
            tokenizer: Arc::new(Tokenizer::from_file(&options.tokenizer_path).unwrap()),
            stop_words: StopWords::default(),
        }
    }
}

#[async_trait]
impl TextGeneration for LlamaEngine {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let cancel = CancellationToken::new();
        let cancel_for_inference = cancel.clone();
        let _guard = cancel.drop_guard();

        let prompt = prompt.to_owned();
        let engine = self.engine.clone();
        let mut stop_condition = self
            .stop_words
            .create_condition(self.tokenizer.clone(), options.stop_words);

        let output_ids = tokio::task::spawn_blocking(move || {
            let engine = engine.lock().unwrap();
            let mut next_token_id = engine.start(&prompt);
            let mut n_remains = options.max_decoding_length - 1;
            let mut output_ids = vec![next_token_id];

            while n_remains > 0 {
                if cancel_for_inference.is_cancelled() {
                    // The token was cancelled
                    break
                }

                next_token_id = engine.step(next_token_id);
                if stop_condition.next_token(next_token_id) {
                    break;
                }
                output_ids.push(next_token_id);
                n_remains -= 1;
            }

            output_ids
        })
        .await
        .expect("Inference failed");

        self.tokenizer.decode(&output_ids, true).unwrap()
    }
}
