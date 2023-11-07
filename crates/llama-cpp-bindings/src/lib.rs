mod llama;
mod utils;

use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use ffi::create_engine;
use futures::stream::BoxStream;
use llama::LlamaService;
use tabby_inference::{
    decoding::StopConditionFactory, helpers, TextGeneration, TextGenerationOptions,
};

#[cxx::bridge(namespace = "llama")]
mod ffi {
    struct StepOutput {
        request_id: u32,
        text: String,
    }

    unsafe extern "C++" {
        include!("llama-cpp-bindings/include/engine.h");

        type TextInferenceEngine;

        fn create_engine(use_gpu: bool, model_path: &str) -> UniquePtr<TextInferenceEngine>;

        fn add_request(
            self: Pin<&mut TextInferenceEngine>,
            request_id: u32,
            prompt: &str,
            max_input_length: usize,
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
}

pub struct LlamaTextGeneration {
    service: LlamaService,
    stop_condition_factory: StopConditionFactory,
}

impl LlamaTextGeneration {
    pub fn new(options: LlamaTextGenerationOptions) -> Self {
        let engine = create_engine(options.use_gpu, &options.model_path);
        if engine.is_null() {
            fatal!("Unable to load model: {}", options.model_path);
        }

        Self {
            service: LlamaService::new(engine),
            stop_condition_factory: StopConditionFactory::default(),
        }
    }
}

#[async_trait]
impl TextGeneration for LlamaTextGeneration {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let s = self.generate_stream(prompt, options).await;
        helpers::stream_to_string(s).await
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String> {
        let stop_condition = self.stop_condition_factory.create(prompt, options.language);

        let mut rx = self
            .service
            .add_request(prompt, options.max_input_length, stop_condition)
            .await;

        let s = stream! {
            let mut length = 0;
            while let Some(new_text) = rx.recv().await {
                yield new_text;
                length += 1;
                if length >= options.max_decoding_length {
                    break;
                }
            }

            rx.close();
        };

        Box::pin(s)
    }
}
