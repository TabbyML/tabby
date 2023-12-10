use async_trait::async_trait;
use tabby_common::api::{
    accelerator::Accelerator,
    code::{CodeSearch, CodeSearchError, SearchResponse},
    event::RawEventLogger,
};
use tokio_tungstenite::connect_async;

pub use crate::{
    schema::worker::{RegisterWorkerError, Worker, WorkerKind},
    websocket::WebSocketTransport,
};

#[tarpc::service]
pub trait Hub {
    async fn register_worker(
        kind: WorkerKind,
        port: i32,
        name: String,
        device: String,
        arch: String,
        cpu_info: String,
        cpu_count: i32,
        accelerators: Vec<Accelerator>,
        token: String,
    ) -> Result<Worker, RegisterWorkerError>;

    async fn log_event(content: String);

    async fn search(q: String, limit: usize, offset: usize) -> SearchResponse;

    async fn search_in_language(
        language: String,
        tokens: Vec<String>,
        limit: usize,
        offset: usize,
    ) -> SearchResponse;
}

pub fn tracing_context() -> tarpc::context::Context {
    tarpc::context::current()
}

pub async fn create_client(addr: &str) -> HubClient {
    let addr = format!("ws://{}/hub", addr);
    let (socket, _) = connect_async(&addr).await.unwrap();
    HubClient::new(Default::default(), WebSocketTransport::from(socket)).spawn()
}

impl RawEventLogger for HubClient {
    fn log(&self, content: String) {
        let context = tarpc::context::current();
        let client = self.clone();

        tokio::spawn(async move { client.log_event(context, content).await });
    }
}

#[async_trait]
impl CodeSearch for HubClient {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        match self
            .search(tracing_context(), q.to_owned(), limit, offset)
            .await
        {
            Ok(serp) => Ok(serp),
            Err(_) => Err(CodeSearchError::NotReady),
        }
    }

    async fn search_in_language(
        &self,
        language: &str,
        tokens: &[String],
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        match self
            .search_in_language(
                tracing_context(),
                language.to_owned(),
                tokens.to_owned(),
                limit,
                offset,
            )
            .await
        {
            Ok(serp) => Ok(serp),
            Err(_) => Err(CodeSearchError::NotReady),
        }
    }
}
