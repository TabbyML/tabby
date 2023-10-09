use std::sync::Arc;

use async_trait::async_trait;
use derive_builder::Builder;
use futures::stream::BoxStream;
use tabby_inference::{
    decoding::{DecodingFactory, IncrementalDecoding},
    helpers, TextGeneration, TextGenerationOptions,
};
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
}

pub struct InferenceContext {
    decoding: IncrementalDecoding,
    cancel: CancellationToken,
}

impl InferenceContext {
    fn new(decoding: IncrementalDecoding, cancel: CancellationToken) -> Self {
        InferenceContext { decoding, cancel }
    }
}

pub struct CTranslate2Engine {
    engine: cxx::SharedPtr<ffi::TextInferenceEngine>,
    decoding_factory: DecodingFactory,
    tokenizer: Arc<Tokenizer>,
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
            decoding_factory: DecodingFactory::default(),
            tokenizer: Arc::new(Tokenizer::from_file(&options.tokenizer_path).unwrap()),
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

        let decoding = self.decoding_factory.create_incremental_decoding(
            self.tokenizer.clone(),
            truncate_tokens(encoding.get_ids(), options.max_input_length),
            options.stop_words,
        );

        let context = InferenceContext::new(decoding, cancel_for_inference);
        let output_ids = tokio::task::spawn_blocking(move || {
            let context = Box::new(context);
            engine.inference(
                context,
                inference_callback,
                truncate_tokens(encoding.get_tokens(), options.max_input_length),
                options.max_decoding_length,
                options.sampling_temperature,
            )
        })
        .await
        .expect("Inference failed");
        self.tokenizer.decode(&output_ids, true).unwrap()
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String> {
        helpers::string_to_stream(self.generate(prompt, options).await).await
    }
}

fn truncate_tokens<T>(tokens: &[T], max_length: usize) -> &[T] {
    if max_length < tokens.len() {
        let start = tokens.len() - max_length;
        &tokens[start..]
    } else {
        tokens
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
    } else if let Some(_) = context.decoding.next_token(token_id) {
        false
    } else {
        true
    }
}
