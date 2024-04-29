//! Bindings to raw C++ LLaMA implementation. implements the TextGeneration trait and being used in tabby to generate text / code.
mod llama;
mod utils;

use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use ffi::create_engine;
use futures::stream::BoxStream;
use llama::{LlamaInitRequest, LlamaService};
use tabby_inference::{CompletionOptions, CompletionStream};

#[cxx::bridge(namespace = "llama")]
mod ffi {
    extern "Rust" {
        type LlamaInitRequest;
        fn id(&self) -> u32;
        fn prompt(&self) -> &str;
        fn max_input_length(&self) -> usize;
        fn seed(&self) -> u64;
        fn temperature(&self) -> f32;

        fn step(&self, token: &str) -> bool;
    }

    unsafe extern "C++" {
        include!("llama-cpp-bindings/include/engine.h");

        type TextInferenceEngine;

        fn create_engine(
            use_gpu: bool,
            model_path: &str,
            parallelism: u8,
        ) -> UniquePtr<TextInferenceEngine>;

        fn add_request(self: Pin<&mut TextInferenceEngine>, context: Box<LlamaInitRequest>);
        fn has_pending_requests(&self) -> bool;
        fn step(self: Pin<&mut TextInferenceEngine>) -> Result<()>;
    }
}

unsafe impl Send for ffi::TextInferenceEngine {}
unsafe impl Sync for ffi::TextInferenceEngine {}

#[derive(Builder, Debug)]
pub struct LlamaTextGenerationOptions {
    model_path: String,
    use_gpu: bool,
    parallelism: u8,
}

pub struct LlamaTextGeneration {
    service: LlamaService,
}

impl LlamaTextGeneration {
    pub fn new(options: LlamaTextGenerationOptions) -> Self {
        let engine = create_engine(options.use_gpu, &options.model_path, options.parallelism);
        if engine.is_null() {
            fatal!("Unable to load model: {}", options.model_path);
        }

        Self {
            service: LlamaService::new(engine),
        }
    }
}

#[async_trait]
impl CompletionStream for LlamaTextGeneration {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
        let mut output_token_budget = options.max_decoding_tokens;
        let mut rx = self
            .service
            .add_request(
                prompt,
                options.max_input_length,
                options.sampling_temperature,
                options.seed,
            )
            .await;

        let s = stream! {
            while let Some(new_text) = rx.recv().await {
                yield new_text;
                output_token_budget -= 1;
                if output_token_budget <= 0 {
                    break;
                }
            }

            rx.close();
        };

        Box::pin(s)
    }
}
