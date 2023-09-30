use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use ffi::create_engine;
use futures::{lock::Mutex, stream::BoxStream};
use tabby_inference::{decoding::DecodingFactory, helpers, TextGeneration, TextGenerationOptions};
use tokenizers::tokenizer::Tokenizer;

#[cxx::bridge(namespace = "llama")]
mod ffi {
    unsafe extern "C++" {
        include!("llama-cpp-bindings/include/engine.h");

        type TextInferenceEngine;

        fn create_engine(model_path: &str) -> UniquePtr<TextInferenceEngine>;

        fn start(self: Pin<&mut TextInferenceEngine>, input_token_ids: &[u32]);
        fn step(self: Pin<&mut TextInferenceEngine>) -> u32;
        fn end(self: Pin<&mut TextInferenceEngine>);

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
    engine: Mutex<cxx::UniquePtr<ffi::TextInferenceEngine>>,
    tokenizer: Arc<Tokenizer>,
    decoding_factory: DecodingFactory,
}

impl LlamaEngine {
    pub fn create(options: LlamaEngineOptions) -> Self {
        LlamaEngine {
            engine: Mutex::new(create_engine(&options.model_path)),
            tokenizer: Arc::new(Tokenizer::from_file(&options.tokenizer_path).unwrap()),
            decoding_factory: DecodingFactory::default(),
        }
    }
}

#[async_trait]
impl TextGeneration for LlamaEngine {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let s = self.generate_stream(prompt, options).await;
        helpers::stream_to_string(s).await
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String> {
        let encoding = self.tokenizer.encode(prompt, true).unwrap();

        let s = stream! {
            let mut engine = self.engine.lock().await;
            let mut engine = engine.as_mut().unwrap();
            let eos_token = engine.eos_token();

            let input_token_ids = truncate_tokens(encoding.get_ids(), options.max_input_length);
            engine.as_mut().start(input_token_ids);
            let mut decoding = self.decoding_factory.create(self.tokenizer.clone(), input_token_ids, &options.stop_words, options.static_stop_words);
            let mut n_remains = options.max_decoding_length ;
            while n_remains > 0 {
                let next_token_id = engine.as_mut().step();
                if next_token_id == eos_token {
                    break;
                }

                if let Some(new_text) = decoding.next_token(next_token_id) {
                    yield new_text;
                } else {
                    break;
                }

                n_remains -= 1;
            }

            engine.end();
        };

        Box::pin(s)
    }
}

fn truncate_tokens(tokens: &[u32], max_length: usize) -> &[u32] {
    if max_length < tokens.len() {
        let start = tokens.len() - max_length;
        &tokens[start..]
    } else {
        tokens
    }
}
