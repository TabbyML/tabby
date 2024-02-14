use std::thread::JoinHandle;

use cxx::UniquePtr;
use tokio::sync::{
    mpsc::{channel, unbounded_channel, Receiver, Sender, UnboundedReceiver, UnboundedSender},
    RwLock,
};

use crate::ffi;

pub(crate) struct LlamaInitRequest {
    id: u32,
    prompt: String,
    max_input_length: usize,
    temperature: f32,
    seed: u64,

    tx: UnboundedSender<String>,
}

impl LlamaInitRequest {
    pub(crate) fn id(&self) -> u32 {
        self.id
    }

    pub(crate) fn prompt(&self) -> &str {
        &self.prompt
    }

    pub(crate) fn max_input_length(&self) -> usize {
        self.max_input_length
    }

    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }

    pub(crate) fn temperature(&self) -> f32 {
        self.temperature
    }

    pub(crate) fn check_candidate(&self, _candidate: &[u8]) -> bool {
        true
    }

    pub(crate) fn accept_candidate(&self, _candidate: &[u8]) {}

    pub(crate) fn step(&self, token: &str) -> bool {
        self.tx.send(token.to_owned()).is_err()
    }
}

struct LlamaServiceImpl {
    engine: cxx::UniquePtr<ffi::TextInferenceEngine>,
    rx: Receiver<LlamaInitRequest>,
}

impl LlamaServiceImpl {
    fn new(engine: UniquePtr<ffi::TextInferenceEngine>, rx: Receiver<LlamaInitRequest>) -> Self {
        Self { engine, rx }
    }

    async fn background_job(&mut self) {
        while let Some(req) = {
            if self.engine.as_ref().unwrap().has_pending_requests() {
                self.rx.try_recv().ok()
            } else {
                self.rx.recv().await
            }
        } {
            // Drop canceled requests.
            if req.tx.is_closed() {
                continue;
            }

            self.engine.as_mut().unwrap().add_request(Box::new(req));
        }

        match self.engine.as_mut().unwrap().step() {
            Ok(()) => (),
            Err(err) => {
                crate::fatal!("Failed to step: {}", err)
            }
        };
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
    next_request_id: RwLock<u32>,
}

impl LlamaService {
    pub fn new(engine: UniquePtr<ffi::TextInferenceEngine>) -> Self {
        let (tx, rx) = channel(1);
        start_llama_service_impl(engine, rx);
        Self {
            tx,
            next_request_id: RwLock::default(),
        }
    }

    async fn alloc_request_id(&self) -> u32 {
        let ret = *self.next_request_id.read().await;

        // 1048576 (2^20) should be large enough to avoid any collision.
        // request_id must be unique at time of processing and shouldn't overflow, tabby is unlikely to handle 2^20 requests at once
        *self.next_request_id.write().await = (ret + 1) % 1048576;
        ret
    }

    pub async fn add_request(
        &self,
        prompt: &str,
        max_input_length: usize,
        temperature: f32,
        seed: u64,
    ) -> UnboundedReceiver<String> {
        let id = self.alloc_request_id().await;
        let (tx, rx) = unbounded_channel();
        self.tx
            .send(LlamaInitRequest {
                id,
                prompt: prompt.to_owned(),
                temperature,
                seed,
                tx,
                max_input_length,
            })
            .await
            .expect("Failed to add request");

        rx
    }
}
