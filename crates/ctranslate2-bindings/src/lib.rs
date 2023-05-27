use tokenizers::tokenizer::{Tokenizer};

#[macro_use]
extern crate derive_builder;

#[cxx::bridge(namespace = "tabby")]
mod ffi {
    unsafe extern "C++" {
        include!("ctranslate2-bindings/include/ctranslate2.h");

        type TextInferenceEngine;

        fn create_engine(
            model_path: &str,
            model_type: &str,
            device: &str,
            device_indices: &[i32],
            num_replicas_per_device: usize,
        ) -> UniquePtr<TextInferenceEngine>;

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
pub struct TextInferenceEngineCreateOptions {
    model_path: String,

    model_type: String,

    tokenizer_path: String,

    device: String,

    device_indices: Vec<i32>,

    num_replicas_per_device: usize,
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
    engine: cxx::UniquePtr<ffi::TextInferenceEngine>,
    tokenizer: Tokenizer,
}

unsafe impl Send for TextInferenceEngine {}
unsafe impl Sync for TextInferenceEngine {}

impl TextInferenceEngine {
    pub fn create(options: TextInferenceEngineCreateOptions) -> Self where {
        let engine = ffi::create_engine(
            &options.model_path,
            &options.model_type,
            &options.device,
            &options.device_indices,
            options.num_replicas_per_device,
        );
        return TextInferenceEngine {
            engine: engine,
            tokenizer: Tokenizer::from_file(&options.tokenizer_path).unwrap(),
        };
    }

    pub fn inference(&self, prompt: &str, options: TextInferenceOptions) -> String {
        let encoding = self.tokenizer.encode(prompt, true).unwrap();
        let output_tokens = self.engine.inference(
            encoding.get_tokens(),
            options.max_decoding_length,
            options.sampling_temperature,
            options.beam_size,
        );
        let output_ids: Vec<u32> = output_tokens
            .iter()
            .filter_map(|x| {
                match self.tokenizer.token_to_id(x) {
                    Some(y) => Some(y),
                    None => { println!("Warning: token ({}) missed in vocab", x); None }
                }
            })
            .collect();
        self.tokenizer.decode(output_ids, true).unwrap()
    }
}
