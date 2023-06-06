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

    stop_words: &'static Vec<&'static str>
}

pub struct InferenceContext {
    stop_re: Option<Regex>,
    cancel: CancellationToken,
    output_text: String
}

impl InferenceContext {
    fn new(stop_re: Option<Regex>, cancel: CancellationToken) -> Self {
        InferenceContext { stop_re, cancel, output_text: "".to_owned() }
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

        let stop_re = if options.stop_words.is_empty() {
            None
        } else {
            let encodings = self.tokenizer.encode_batch(options.stop_words.clone(), false).unwrap();
            let stop_tokens : Vec<String> = encodings.iter().map(|x| x.get_tokens().join("")).collect();
            let regex_string = r"(?m)".to_owned() + &stop_tokens.join("|");
            Some(Regex::new(&regex_string).unwrap())
        };

        let context = InferenceContext::new(stop_re, cancel_for_inference);
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
        let output_text = self.tokenizer.decode(output_ids, true).unwrap();
        for stop_word in options.stop_words {
            if let Some(stripped_text) = output_text.strip_suffix(stop_word) {
                return stripped_text.to_string();
            }
        }
        output_text
    }
}

fn inference_callback(context: &mut InferenceContext, _step: usize, _token_id: u32, token: String) -> bool {
    if context.cancel.is_cancelled() {
        true
    } else {
        if let Some(re) = &context.stop_re {
            context.output_text.push_str(&token);
            re.find(&context.output_text).is_some()
        } else {
            false
        }
    }
}
