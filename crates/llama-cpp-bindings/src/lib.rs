use std::{collections::HashMap, sync::Arc};

use async_stream::stream;
use async_trait::async_trait;
use cxx::UniquePtr;
use derive_builder::Builder;
use ffi::create_engine;
use futures::{lock::Mutex, stream::BoxStream};
use tabby_inference::{
    decoding::{StopCondition, StopConditionFactory},
    helpers, TextGeneration, TextGenerationOptions,
};
use tokio::{
    sync::mpsc::{channel, Sender},
    task::yield_now,
};

#[cxx::bridge(namespace = "llama")]
mod ffi {
    struct StepOutput {
        request_id: u32,
        text: String,
    }

    unsafe extern "C++" {
        include!("llama-cpp-bindings/include/engine.h");

        type TextInferenceEngine;

        fn create_engine(use_gpu: bool, model_path: &str) -> UniquePtr<TextInferenceEngine>;

        fn add_request(
            self: Pin<&mut TextInferenceEngine>,
            request_id: u32,
            prompt: &str,
            max_input_length: usize,
        );
        fn stop_request(self: Pin<&mut TextInferenceEngine>, request_id: u32);
        fn step(self: Pin<&mut TextInferenceEngine>) -> Result<Vec<StepOutput>>;
    }
}

unsafe impl Send for ffi::TextInferenceEngine {}
unsafe impl Sync for ffi::TextInferenceEngine {}

struct InferenceRequest {
    tx: Sender<String>,
    stop_condition: StopCondition,
}

struct AsyncTextInferenceEngine {
    engine: Mutex<cxx::UniquePtr<ffi::TextInferenceEngine>>,
    stop_condition_factory: StopConditionFactory,
    requests: Mutex<HashMap<u32, InferenceRequest>>,

    next_request_id: Mutex<u32>,
}

impl AsyncTextInferenceEngine {
    fn create(engine: UniquePtr<ffi::TextInferenceEngine>) -> Self {
        Self {
            engine: Mutex::new(engine),
            stop_condition_factory: StopConditionFactory::default(),
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

        let result = match engine.as_mut().unwrap().step() {
            Ok(result) => result,
            Err(err) => panic!("Failed to step: {}", err),
        };

        for ffi::StepOutput { request_id, text } in result {
            let mut stopped = false;
            let InferenceRequest { tx, stop_condition } = requests.get_mut(&request_id).unwrap();

            if tx.is_closed() || text.is_empty() {
                // Cancelled by client side or hit eos.
                stopped = true;
            } else if !stop_condition.should_stop(&text) {
                match tx.send(text).await {
                    Ok(_) => (),
                    Err(_) => stopped = true,
                }
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
        let stop_condition = self.stop_condition_factory.create(prompt, options.language);

        let (tx, mut rx) = channel::<String>(4);
        {
            let mut engine = self.engine.lock().await;

            let mut request_id = self.next_request_id.lock().await;
            self.requests
                .lock()
                .await
                .insert(*request_id, InferenceRequest { tx, stop_condition });
            engine
                .as_mut()
                .unwrap()
                .add_request(*request_id, prompt, options.max_input_length);

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
        let ret = LlamaTextGeneration {
            engine: Arc::new(AsyncTextInferenceEngine::create(engine)),
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
