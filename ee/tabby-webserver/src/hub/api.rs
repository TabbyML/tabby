use async_trait::async_trait;
use axum::{headers::Header, http::HeaderName};
use hyper::Request;
use serde::{Deserialize, Serialize};
use tabby_common::api::{
    code::{CodeSearch, CodeSearchError, SearchResponse},
    event::RawEventLogger,
};
use tokio_tungstenite::connect_async;

use super::websocket::WebSocketTransport;
pub use crate::schema::worker::WorkerKind;

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

pub async fn create_client(addr: &str, token: &str, request: RegisterWorkerRequest) -> HubClient {
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
            serde_json::to_string(&request).unwrap(),
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

#[derive(Serialize, Deserialize)]
pub struct RegisterWorkerRequest {
    pub kind: WorkerKind,
    pub port: i32,
    pub name: String,
    pub device: String,
    pub arch: String,
    pub cpu_info: String,
    pub cpu_count: i32,
    pub cuda_devices: Vec<String>,
}

pub static REGISTER_WORKER_HEADER: HeaderName = HeaderName::from_static("x-tabby-register-worker");

impl Header for RegisterWorkerRequest {
    fn name() -> &'static axum::http::HeaderName {
        &REGISTER_WORKER_HEADER
    }

    fn decode<'i, I>(values: &mut I) -> Result<Self, axum::headers::Error>
    where
        Self: Sized,
        I: Iterator<Item = &'i axum::http::HeaderValue>,
    {
        let mut x: Vec<_> = values
            .map(|x| serde_json::from_slice(x.as_bytes()))
            .collect();
        if let Some(x) = x.pop() {
            x.map_err(|_| axum::headers::Error::invalid())
        } else {
            Err(axum::headers::Error::invalid())
        }
    }

    fn encode<E: Extend<axum::http::HeaderValue>>(&self, _values: &mut E) {
        todo!()
    }
}
