use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use futures::stream::BoxStream;
use stop_words::{IncrementalDecoding, DecodingFactory};
use tabby_inference::{helpers, TextGeneration, TextGenerationOptions};
use tokenizers::tokenizer::Tokenizer;
use tokio::sync::mpsc::{channel, Sender};
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
    sender: Sender<String>,
    decoding: IncrementalDecoding,
    cancel: CancellationToken,
}

impl InferenceContext {
    fn new(
        sender: Sender<String>,
        decoding: IncrementalDecoding,
        cancel: CancellationToken,
    ) -> Self {
        InferenceContext {
            sender,
            decoding,
            cancel,
        }
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
        let s = self.generate_stream(prompt, options).await;
        helpers::stream_to_string(s).await
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String> {
        let encoding = self.tokenizer.encode(prompt, true).unwrap();
        let engine = self.engine.clone();
        let s = stream! {
            let cancel = CancellationToken::new();
            let cancel_for_inference = cancel.clone();
            let _guard = cancel.drop_guard();

            let decoding = self
                .decoding_factory
                .create_incremental_decoding(self.tokenizer.clone(), encoding.get_ids(), options.stop_words);

            let (sender, mut receiver) = channel::<String>(8);
            let context = InferenceContext::new(sender, decoding, cancel_for_inference);
            tokio::task::spawn(async move {
                let context = Box::new(context);
                engine.inference(
                    context,
                    inference_callback,
                    truncate_tokens(encoding.get_tokens(), options.max_input_length),
                    options.max_decoding_length,
                    options.sampling_temperature,
                    );
            });

            while let Some(text) = receiver.recv().await {
                yield text;
            }
        };
        Box::pin(s)
    }
}

fn truncate_tokens(tokens: &[String], max_length: usize) -> &[String] {
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
    } else if let Some(new_text) = context.decoding.next_token(token_id) {
        let _ = context.sender.blocking_send(new_text);
        false
    } else {
        true
    }
}
