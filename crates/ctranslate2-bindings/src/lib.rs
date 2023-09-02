use async_trait::async_trait;
use derive_builder::Builder;
use stop_words::{StopWords, StopWordsCondition};
use tabby_inference::{TextGeneration, TextGenerationOptions};
use tokenizers::tokenizer::Tokenizer;
use tokio_util::sync::CancellationToken;

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
pub struct CTranslate2EngineOptions {
    model_path: String,

    model_type: String,

    tokenizer_path: String,

    device: String,

    device_indices: Vec<i32>,

    num_replicas_per_device: usize,

    compute_type: String,

    stop_words_encoding_offset: Option<usize>,
}

pub struct InferenceContext {
    stop_condition: StopWordsCondition,
    cancel: CancellationToken,
}

impl InferenceContext {
    fn new(stop_condition: StopWordsCondition, cancel: CancellationToken) -> Self {
        InferenceContext {
            stop_condition,
            cancel,
        }
    }
}

pub struct CTranslate2Engine {
    engine: cxx::SharedPtr<ffi::TextInferenceEngine>,
    stop_words: StopWords,
    tokenizer: Tokenizer,
    stop_words_encoding_offset: Option<usize>,
}

impl CTranslate2Engine {
    pub fn create(options: CTranslate2EngineOptions) -> Self where {
        let engine = ffi::create_engine(
            &options.model_path,
            &options.model_type,
            &options.device,
            &options.compute_type,
            &options.device_indices,
            options.num_replicas_per_device,
        );

        return Self {
            engine,
            stop_words: StopWords::new(),
            tokenizer: Tokenizer::from_file(&options.tokenizer_path).unwrap(),
            stop_words_encoding_offset: options.stop_words_encoding_offset,
        };
    }
}

#[async_trait]
impl TextGeneration for CTranslate2Engine {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let encoding = self.tokenizer.encode(prompt, true).unwrap();
        let engine = self.engine.clone();

        let cancel = CancellationToken::new();
        let cancel_for_inference = cancel.clone();
        let _guard = cancel.drop_guard();

        let stop_condition = self.stop_words.create_condition(
            &self.tokenizer,
            options.stop_words,
            self.stop_words_encoding_offset,
        );
        let context = InferenceContext::new(stop_condition, cancel_for_inference);
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
    token_id: u32,
    _token: String,
) -> bool {
    if context.cancel.is_cancelled() {
        true
    } else {
        context.stop_condition.next_token(token_id)
    }
}
