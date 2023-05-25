use std::sync::Mutex;
use tokenizers::tokenizer::{Model, Tokenizer};

#[macro_use]
extern crate derive_builder;

#[cxx::bridge(namespace = "tabby")]
mod ffi {
    unsafe extern "C++" {
        include!("ctranslate2-bindings/include/ctranslate2.h");

        type TextInferenceEngine;

        fn create_engine(model_path: &str) -> UniquePtr<TextInferenceEngine>;
        fn inference(
            &self,
            tokens: &[String],
            max_decoding_length: usize,
            sampling_temperature: f32,
            beam_size: usize,
        ) -> Vec<String>;
    }
}

#[derive(Builder, Debug)]
pub struct TextInferenceOptions {
    #[builder(default = "256")]
    max_decoding_length: usize,

    #[builder(default = "1.0")]
    sampling_temperature: f32,

    #[builder(default = "2")]
    beam_size: usize,
}

pub struct TextInferenceEngine {
    engine: Mutex<cxx::UniquePtr<ffi::TextInferenceEngine>>,
    tokenizer: Tokenizer,
}

unsafe impl Send for TextInferenceEngine {}
unsafe impl Sync for TextInferenceEngine {}

impl TextInferenceEngine {
    pub fn create(model_path: &str, tokenizer_path: &str) -> Self where {
        return TextInferenceEngine {
            engine: Mutex::new(ffi::create_engine(model_path)),
            tokenizer: Tokenizer::from_file(tokenizer_path).unwrap(),
        };
    }

    pub fn inference(&self, prompt: &str, options: TextInferenceOptions) -> String {
        let encoding = self.tokenizer.encode(prompt, true).unwrap();
        let output_tokens = self.engine.lock().unwrap().inference(
            encoding.get_tokens(),
            options.max_decoding_length,
            options.sampling_temperature,
            options.beam_size,
        );

        let model = self.tokenizer.get_model();
        let output_ids: Vec<u32> = output_tokens
            .iter()
            .map(|x| model.token_to_id(x).unwrap())
            .collect();
        self.tokenizer.decode(output_ids, true).unwrap()
    }
}
