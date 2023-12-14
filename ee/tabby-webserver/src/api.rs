use async_trait::async_trait;
use hyper::{Request};
use tabby_common::api::{
    code::{CodeSearch, CodeSearchError, SearchResponse},
    event::RawEventLogger,
};
use tokio_tungstenite::{connect_async};

pub use crate::schema::worker::{RegisterWorkerError, Worker, WorkerKind};
use crate::{websocket::WebSocketTransport, RegisterWorkerRequest, REGISTER_WORKER_HEADER};

#[tarpc::service]
pub trait Hub {
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

pub async fn create_client(
    addr: &str,
    token: &str,
    kind: WorkerKind,
    port: i32,
    name: String,
    device: String,
    arch: String,
    cpu_info: String,
    cpu_count: i32,
    cuda_devices: Vec<String>,
) -> HubClient {
    let request = Request::builder()
        .uri(format!("ws://{}/hub", addr))
        .header("Host", addr)
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header("Sec-WebSocket-Key", "unused")
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .header(
            &REGISTER_WORKER_HEADER,
            serde_json::to_string(&RegisterWorkerRequest {
                kind,
                port,
                name,
                device,
                arch,
                cpu_info,
                cpu_count,
                cuda_devices,
            })
            .unwrap(),
        )
        .body(())
        .unwrap();

    let (socket, _) = connect_async(request).await.unwrap();
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
