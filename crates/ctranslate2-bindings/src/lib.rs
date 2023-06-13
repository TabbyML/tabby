use dashmap::DashMap;
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
            compute_type: &str,
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
        ) -> Vec<u32>;
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

    compute_type: String,
}

#[derive(Builder, Debug)]
pub struct TextInferenceOptions {
    #[builder(default = "256")]
    max_decoding_length: usize,

    #[builder(default = "1.0")]
    sampling_temperature: f32,

    stop_words: &'static Vec<&'static str>,
}

pub struct InferenceContext {
    stop_re: Option<Regex>,
    cancel: CancellationToken,
    reversed_output_text: String,
}

impl InferenceContext {
    fn new(stop_re: Option<Regex>, cancel: CancellationToken) -> Self {
        InferenceContext {
            stop_re,
            cancel,
            reversed_output_text: "".to_owned(),
        }
    }
}

pub struct TextInferenceEngine {
    engine: cxx::SharedPtr<ffi::TextInferenceEngine>,
    tokenizer: Tokenizer,
    stop_regex_cache: DashMap<&'static Vec<&'static str>, Regex>,
}

impl TextInferenceEngine {
    pub fn create(options: TextInferenceEngineCreateOptions) -> Self where {
        let engine = ffi::create_engine(
            &options.model_path,
            &options.model_type,
            &options.device,
            &options.compute_type,
            &options.device_indices,
            options.num_replicas_per_device,
        );
        return TextInferenceEngine {
            engine,
            stop_regex_cache: DashMap::new(),
            tokenizer: Tokenizer::from_file(&options.tokenizer_path).unwrap(),
        };
    }

    pub async fn inference(&self, prompt: &str, options: TextInferenceOptions) -> String {
        let encoding = self.tokenizer.encode(prompt, true).unwrap();
        let engine = self.engine.clone();

        let cancel = CancellationToken::new();
        let cancel_for_inference = cancel.clone();
        let _guard = cancel.drop_guard();

        let stop_re: Option<Regex> = if options.stop_words.is_empty() {
            None
        } else {
            let mut re = self.stop_regex_cache.get(options.stop_words);
            if re.is_none() {
                self.stop_regex_cache.insert(
                    options.stop_words,
                    create_stop_regex(&self.tokenizer, options.stop_words),
                );
                re = self.stop_regex_cache.get(options.stop_words);
            }
            re.map(|x| x.value().clone())
        };

        let context = InferenceContext::new(stop_re, cancel_for_inference);
        let output_ids = tokio::task::spawn_blocking(move || {
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
        self.tokenizer.decode(output_ids, true).unwrap()
    }
}

fn inference_callback(
    context: &mut InferenceContext,
    _step: usize,
    _token_id: u32,
    token: String,
) -> bool {
    if context.cancel.is_cancelled() {
        true
    } else if let Some(re) = &context.stop_re {
        let mut new_token = reverse(token);
        new_token.push_str(&context.reversed_output_text);
        context.reversed_output_text = new_token;
        re.find(&context.reversed_output_text).is_some()
    } else {
        false
    }
}

fn reverse(s: String) -> String {
    s.chars().rev().collect()
}

fn create_stop_regex(tokenizer: &Tokenizer, stop_words: &[&str]) -> Regex {
    let encodings = tokenizer
        .encode_batch(stop_words.to_owned(), false)
        .unwrap();
    let stop_tokens: Vec<String> = encodings
        .iter()
        .map(|x| x.get_tokens().join(""))
        // Reverse for efficient suffix matching.
        .map(reverse)
        .collect();

    // (?m) enables multi-line matching mode.
    // \A means absolute begins of string.
    let regex_string = r"(?m)\A".to_owned() + &stop_tokens.join("|");
    Regex::new(&regex_string).unwrap()
}
