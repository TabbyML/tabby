//! Bindings to raw C++ LLaMA implementation. implements the TextGeneration trait and being used in tabby to generate text / code.
mod llama;
mod utils;

use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use ffi::create_engine;
use futures::stream::BoxStream;
use llama::LlamaService;
use tabby_inference::{TextGenerationOptions, TextGenerationStream};

struct RequestContext {
    id: u32,
    max_input_length: usize,
    temperature: f32,
    seed: u64,
}

impl RequestContext {
    fn id(&self) -> u32 {
        self.id
    }
    fn max_input_length(&self) -> usize {
        self.max_input_length
    }
    fn seed(&self) -> u64 {
        self.seed
    }
    fn temperature(&self) -> f32 {
        self.temperature
    }

    fn check_candidate(&self, _candidate: &[u8]) -> bool {
        true
    }
    fn accept_candidate(&self, _candidate: &[u8]) {}
}

#[cxx::bridge(namespace = "llama")]
mod ffi {
    struct StepOutput {
        request_id: u32,
        text: String,
    }

    extern "Rust" {
        type RequestContext;
        fn id(&self) -> u32;
        fn max_input_length(&self) -> usize;
        fn seed(&self) -> u64;
        fn temperature(&self) -> f32;

        fn check_candidate(&self, candidate: &[u8]) -> bool;
        fn accept_candidate(&self, candidate: &[u8]);
    }

    unsafe extern "C++" {
        include!("llama-cpp-bindings/include/engine.h");

        type TextInferenceEngine;

        fn create_engine(
            use_gpu: bool,
            model_path: &str,
            parallelism: u8,
        ) -> UniquePtr<TextInferenceEngine>;

        fn add_request(
            self: Pin<&mut TextInferenceEngine>,
            context: Box<RequestContext>,
            prompt: &str,
        );
        fn stop_request(self: Pin<&mut TextInferenceEngine>, request_id: u32);
        fn step(self: Pin<&mut TextInferenceEngine>) -> Result<Vec<StepOutput>>;
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
impl TextGenerationStream for LlamaTextGeneration {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> BoxStream<String> {
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
            }

            rx.close();
        };

        Box::pin(s)
    }
}
