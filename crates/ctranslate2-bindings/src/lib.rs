use regex::Regex;
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
            callback: fn(
                &mut InferenceContext,
                // step
                usize,
                // token_id
                u32,
                // token
                String,
            ) -> bool,
            tokens: &[String],
            max_decoding_length: usize,
            sampling_temperature: f32,
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

    #[builder(default = "vec!()")]
    stop_words: Vec<String>
}

pub struct InferenceContext {
    stop_regexp: Regex,
    cancel: CancellationToken,
    output_text: String
}

impl InferenceContext {
    fn new(stop_words: Vec<String>, cancel: CancellationToken) -> Self {
        let stop_regexp = Regex::new(stop_words.join("|").as_ref()).unwrap();
        InferenceContext { stop_regexp, cancel, output_text: "".to_owned() }
    }
}

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

        let context = InferenceContext::new(options.stop_words, cancel_for_inference);
        let output_tokens = tokio::task::spawn_blocking(move || {
            let context = Box::new(context);
            engine.inference(
                context,
                inference_callback,
                encoding.get_tokens(),
                options.max_decoding_length,
                options.sampling_temperature,
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

fn inference_callback(context: &mut InferenceContext, _step: usize, _token_id: u32, token: String) -> bool {
    if context.cancel.is_cancelled() {
        true
    } else {
        context.output_text.push_str(&token);
        if let Some(_) = context.stop_regexp.find(&context.output_text) {
            true
        } else {
            false
        }
    }
}
