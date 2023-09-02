use async_trait::async_trait;
use derive_builder::Builder;
use ffi::create_engine;
use tabby_inference::{TextGeneration, TextGenerationOptions};

#[cxx::bridge(namespace = "tabby")]
mod ffi {
    unsafe extern "C++" {
        include!("llama-cpp-bindings/include/engine.h");

        type TextInferenceEngine;

        fn create_engine(model_path: &str) -> SharedPtr<TextInferenceEngine>;

        fn inference(
            &self,
            prompt: &str,
            max_decoding_length: usize,
            sampling_temperature: f32,
        ) -> Vec<u32>;
    }
}


unsafe impl Send for ffi::TextInferenceEngine {}
unsafe impl Sync for ffi::TextInferenceEngine {}

#[derive(Builder, Debug)]
pub struct LlamaEngineOptions {
    model_path: String,
}

pub struct LlamaEngine {
    engine: cxx::SharedPtr<ffi::TextInferenceEngine>
}

impl LlamaEngine {
    pub fn create(options: LlamaEngineOptions) -> Self {
        LlamaEngine { engine: create_engine(&options.model_path) }
    }
}


#[async_trait]
impl TextGeneration for LlamaEngine {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        self.engine.inference(prompt, options.max_decoding_length, options.sampling_temperature);
        "abc".to_owned()
    }
}
