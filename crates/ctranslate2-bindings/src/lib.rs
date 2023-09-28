use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use derive_builder::Builder;
use futures::stream::BoxStream;
use stop_words::{StopWords, StopWordsCondition};
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
    sender: Sender<u32>,
    stop_condition: StopWordsCondition,
    cancel: CancellationToken,
}

impl InferenceContext {
    fn new(
        sender: Sender<u32>,
        stop_condition: StopWordsCondition,
        cancel: CancellationToken,
    ) -> Self {
        InferenceContext {
            sender,
            stop_condition,
            cancel,
        }
    }
}

pub struct CTranslate2Engine {
    engine: cxx::SharedPtr<ffi::TextInferenceEngine>,
    stop_words: StopWords,
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
            stop_words: StopWords::default(),
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

            let stop_condition = self
                .stop_words
                .create_condition(self.tokenizer.clone(), options.stop_words);

            let (sender, mut receiver) = channel::<u32>(8);
            let context = InferenceContext::new(sender, stop_condition, cancel_for_inference);
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

            while let Some(next_token_id) = receiver.recv().await {
                let text = self.tokenizer.decode(&[next_token_id], true).unwrap();
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
    let _ = context.sender.blocking_send(token_id);
    if context.cancel.is_cancelled() {
        true
    } else {
        context.stop_condition.next_token(token_id)
    }
}
