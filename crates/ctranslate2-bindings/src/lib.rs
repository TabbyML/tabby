use tokenizers::tokenizer::Tokenizer;
use tokio_util::sync::CancellationToken;

#[macro_use]
extern crate derive_builder;

#[cxx::bridge(namespace = "tabby")]
mod ffi {
    extern "Rust" {
        type InferenceContext;
    }

    unsafe extern "C++" {
        include!("ctranslate2-bindings/include/ctranslate2.h");

        type TextInferenceEngine;

        fn create_engine(
            model_path: &str,
            model_type: &str,
            device: &str,
            device_indices: &[i32],
            num_replicas_per_device: usize,
        ) -> SharedPtr<TextInferenceEngine>;

        fn inference(
            &self,
            context: Box<InferenceContext>,
            is_context_cancelled: fn(&InferenceContext) -> bool,
            tokens: &[String],
            max_decoding_length: usize,
            sampling_temperature: f32,
            beam_size: usize,
        ) -> Vec<String>;
    }
}

unsafe impl Send for ffi::TextInferenceEngine {}
unsafe impl Sync for ffi::TextInferenceEngine {}

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

struct InferenceContext(CancellationToken);

pub struct TextInferenceEngine {
    engine: cxx::SharedPtr<ffi::TextInferenceEngine>,
    tokenizer: Tokenizer,
}

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
            engine,
            tokenizer: Tokenizer::from_file(&options.tokenizer_path).unwrap(),
        };
    }

    pub async fn inference(&self, prompt: &str, options: TextInferenceOptions) -> String {
        let encoding = self.tokenizer.encode(prompt, true).unwrap();
        let engine = self.engine.clone();

        let cancel = CancellationToken::new();
        let cancel_for_inference = cancel.clone();
        let _guard = cancel.drop_guard();

        let context = InferenceContext(cancel_for_inference);
        let output_tokens = tokio::task::spawn_blocking(move || {
            let context = Box::new(context);
            engine.inference(
                context,
                |context| context.0.is_cancelled(),
                encoding.get_tokens(),
                options.max_decoding_length,
                options.sampling_temperature,
                options.beam_size,
            )
        })
        .await
        .expect("Inference failed");
        let output_ids: Vec<u32> = output_tokens
            .iter()
            .filter_map(|x| match self.tokenizer.token_to_id(x) {
                Some(y) => Some(y),
                None => {
                    println!("Warning: token ({}) missed in vocab", x);
                    None
                }
            })
            .collect();
        self.tokenizer.decode(output_ids, true).unwrap()
    }
}
