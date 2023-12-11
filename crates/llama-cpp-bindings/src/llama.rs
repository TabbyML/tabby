use std::{collections::HashMap, thread::JoinHandle};

use cxx::UniquePtr;
use tabby_inference::decoding::StopCondition;
use tokio::sync::mpsc::{channel, Receiver, Sender};

use crate::ffi;

struct LlamaInitRequest {
    prompt: String,
    max_input_length: usize,

    tx: Sender<String>,
    stop_condition: StopCondition,
}

struct LlamaRunningRequest {
    tx: Sender<String>,
    stop_condition: StopCondition,
}

struct LlamaServiceImpl {
    next_request_id: u32,
    engine: cxx::UniquePtr<ffi::TextInferenceEngine>,
    rx: Receiver<LlamaInitRequest>,
    requests: HashMap<u32, LlamaRunningRequest>,
}

impl LlamaServiceImpl {
    fn new(engine: UniquePtr<ffi::TextInferenceEngine>, rx: Receiver<LlamaInitRequest>) -> Self {
        Self {
            next_request_id: 0,
            engine,
            rx,
            requests: HashMap::new(),
        }
    }

    fn alloc_request_id(&mut self) -> u32 {
        let ret = self.next_request_id;

        // 1048576 (2^20) should be large enough to avoid any collision.
        // FIXME(meng): figure out a better way.
        self.next_request_id = (self.next_request_id + 1) % 1048576;
        ret
    }

    async fn next_request(&mut self) -> Option<LlamaInitRequest> {
        if self.requests.is_empty() {
            self.rx.recv().await
        } else {
            self.rx.try_recv().ok()
        }
    }

    async fn background_job(&mut self) {
        while let Some(LlamaInitRequest {
            prompt,
            tx,
            max_input_length,
            stop_condition,
        }) = self.next_request().await
        {
            // Drop canceled requests.
            if tx.is_closed() {
                continue;
            }

            let request_id = self.alloc_request_id();
            self.requests
                .insert(request_id, LlamaRunningRequest { tx, stop_condition });
            self.engine
                .as_mut()
                .unwrap()
                .add_request(request_id, &prompt, max_input_length);
        }

        let result = match self.engine.as_mut().unwrap().step() {
            Ok(result) => result,
            Err(err) => {
                crate::fatal!("Failed to step: {}", err)
            }
        };

        for ffi::StepOutput { request_id, text } in result {
            let mut stopped: bool;
            let LlamaRunningRequest { tx, stop_condition } =
                self.requests.get_mut(&request_id).unwrap();

            if tx.is_closed() || text.is_empty() {
                // Cancelled by client side or hit eos.
                stopped = true;
            } else {
                stopped = stop_condition.should_stop(&text);

                match tx.send(text).await {
                    Ok(_) => (),
                    Err(_) => stopped = true,
                }
            }

            if stopped {
                self.requests.remove(&request_id);
                self.engine.as_mut().unwrap().stop_request(request_id);
            }
        }
    }
}

fn start_llama_service_impl(
    engine: UniquePtr<ffi::TextInferenceEngine>,
    rx: Receiver<LlamaInitRequest>,
) -> JoinHandle<()> {
    let mut service = LlamaServiceImpl::new(engine, rx);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    std::thread::spawn(move || {
        let local = tokio::task::LocalSet::new();
        local.spawn_local(async move {
            loop {
                service.background_job().await;
            }
        });

        rt.block_on(local);
    })
}

pub struct LlamaService {
    tx: Sender<LlamaInitRequest>,
}

impl LlamaService {
    pub fn new(engine: UniquePtr<ffi::TextInferenceEngine>) -> Self {
        let (tx, rx) = channel(1);
        start_llama_service_impl(engine, rx);
        Self { tx }
    }

    pub async fn add_request(
        &self,
        prompt: &str,
        max_input_length: usize,
        stop_condition: StopCondition,
    ) -> Receiver<String> {
        let (tx, rx) = channel(8);
        self.tx
            .send(LlamaInitRequest {
                prompt: prompt.to_owned(),
                tx,
                max_input_length,
                stop_condition,
            })
            .await
            .expect("Failed to add request");

        rx
    }
}
