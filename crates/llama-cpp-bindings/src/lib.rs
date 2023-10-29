use std::{collections::HashMap, sync::Arc};

use async_stream::stream;
use async_trait::async_trait;
use cxx::UniquePtr;
use derive_builder::Builder;
use ffi::create_engine;
use futures::{lock::Mutex, stream::BoxStream};
use tabby_inference::{
    decoding::{DecodingFactory, IncrementalDecoding},
    helpers, TextGeneration, TextGenerationOptions,
};
use tokenizers::tokenizer::Tokenizer;
use tokio::{
    sync::mpsc::{channel, Sender},
    task::yield_now,
};

#[cxx::bridge(namespace = "llama")]
mod ffi {
    unsafe extern "C++" {
        include!("llama-cpp-bindings/include/engine.h");

        type TextInferenceEngine;

        fn create_engine(use_gpu: bool, model_path: &str) -> UniquePtr<TextInferenceEngine>;

        fn add_request(
            self: Pin<&mut TextInferenceEngine>,
            request_id: u32,
            input_token_ids: &[u32],
        );
        fn stop_request(self: Pin<&mut TextInferenceEngine>, request_id: u32);
        fn step(self: Pin<&mut TextInferenceEngine>) -> Result<Vec<u32>>;

        fn eos_token_id(&self) -> u32;
    }
}

unsafe impl Send for ffi::TextInferenceEngine {}
unsafe impl Sync for ffi::TextInferenceEngine {}

struct InferenceRequest {
    tx: Sender<String>,
    decoding: IncrementalDecoding,
}

struct AsyncTextInferenceEngine {
    engine: Mutex<cxx::UniquePtr<ffi::TextInferenceEngine>>,
    tokenizer: Arc<Tokenizer>,
    decoding_factory: DecodingFactory,
    requests: Mutex<HashMap<u32, InferenceRequest>>,

    next_request_id: Mutex<u32>,
    eos_token_id: u32,
}

impl AsyncTextInferenceEngine {
    fn create(engine: UniquePtr<ffi::TextInferenceEngine>, tokenizer: Tokenizer) -> Self {
        Self {
            eos_token_id: engine.eos_token_id(),
            engine: Mutex::new(engine),
            tokenizer: Arc::new(tokenizer),
            decoding_factory: DecodingFactory::default(),
            requests: Mutex::new(HashMap::new()),
            next_request_id: Mutex::new(0),
        }
    }

    async fn background_job(&self) {
        let mut requests = self.requests.lock().await;
        if requests.len() == 0 {
            return;
        }

        let mut engine = self.engine.lock().await;

        let Ok(result) = engine.as_mut().unwrap().step() else {
            panic!("Failed to evaluation");
        };

        for i in (0..result.len()).step_by(2) {
            let request_id = result[i];
            let token_id = result[i + 1];

            let InferenceRequest { tx, decoding } = requests.get_mut(&request_id).unwrap();
            let mut stopped = false;

            if tx.is_closed() || token_id == self.eos_token_id {
                // Cancelled by client side or hit eos.
                stopped = true;
            } else if let Some(new_text) = decoding.next_token(token_id) {
                tx.send(new_text).await.expect("send failed");
            } else {
                // Stoop words stopped
                stopped = true;
            }

            if stopped {
                requests.remove(&request_id);
                engine.as_mut().unwrap().stop_request(request_id);
            }
        }
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String> {
        let encoding = self.tokenizer.encode(prompt, true).unwrap();
        let input_token_ids = truncate_tokens(encoding.get_ids(), options.max_input_length);
        let decoding = self.decoding_factory.create_incremental_decoding(
            self.tokenizer.clone(),
            input_token_ids,
            options.language,
        );

        let (tx, mut rx) = channel::<String>(4);
        {
            let mut engine = self.engine.lock().await;
            let engine = engine.as_mut().unwrap();

            let mut request_id = self.next_request_id.lock().await;
            self.requests
                .lock()
                .await
                .insert(*request_id, InferenceRequest { tx, decoding });
            engine.add_request(*request_id, input_token_ids);

            // 2048 should be large enough to avoid collision.
            *request_id = (*request_id + 1) % 2048;
        }

        let s = stream! {
            let mut length = 0;
            while let Some(new_text) = rx.recv().await {
                yield new_text;
                length += 1;
                if length >= options.max_decoding_length {
                    break;
                }
            }

            rx.close();
        };

        Box::pin(s)
    }
}

#[derive(Builder, Debug)]
pub struct LlamaTextGenerationOptions {
    model_path: String,
    tokenizer_path: String,
    use_gpu: bool,
}

pub struct LlamaTextGeneration {
    engine: Arc<AsyncTextInferenceEngine>,
}

impl LlamaTextGeneration {
    pub fn create(options: LlamaTextGenerationOptions) -> Self {
        let engine = create_engine(options.use_gpu, &options.model_path);
        if engine.is_null() {
            panic!("Unable to load model: {}", options.model_path);
        }
        let tokenizer = Tokenizer::from_file(&options.tokenizer_path).unwrap();
        let ret = LlamaTextGeneration {
            engine: Arc::new(AsyncTextInferenceEngine::create(engine, tokenizer)),
        };
        ret.start_background_job();
        ret
    }

    pub fn start_background_job(&self) {
        let engine = self.engine.clone();
        tokio::spawn(async move {
            loop {
                engine.background_job().await;
                yield_now().await;
            }
        });
    }
}

#[async_trait]
impl TextGeneration for LlamaTextGeneration {
    async fn generate(&self, prompt: &str, options: TextGenerationOptions) -> String {
        let s = self.generate_stream(prompt, options).await;
        helpers::stream_to_string(s).await
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        options: TextGenerationOptions,
    ) -> BoxStream<String> {
        self.engine.generate_stream(prompt, options).await
    }
}

fn truncate_tokens(tokens: &[u32], max_length: usize) -> &[u32] {
    if max_length < tokens.len() {
        let start = tokens.len() - max_length;
        &tokens[start..]
    } else {
        tokens
    }
}
