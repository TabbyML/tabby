use std::net::IpAddr;

use anyhow::Result;
use async_trait::async_trait;
use axum::{extract::Request, http::HeaderName};
use axum_extra::headers::Header;
use serde::{Deserialize, Serialize};
use tabby_common::{
    api::{
        code::{CodeSearch, CodeSearchError, SearchResponse},
        event::{EventLogger, LogEntry},
    },
    config::{RepositoryAccess, RepositoryConfig},
};
use tabby_schema::worker::Worker;
pub use tabby_schema::worker::WorkerKind;
use tarpc::context::Context;
use tokio_tungstenite::connect_async;

use crate::axum::websocket::WebSocketTransport;

#[tarpc::service]
pub trait Hub {
    async fn write_log(x: LogEntry);

    async fn search(q: String, limit: usize, offset: usize) -> SearchResponse;

    async fn search_in_language(
        git_url: String,
        language: String,
        tokens: Vec<String>,
        limit: usize,
        offset: usize,
    ) -> SearchResponse;

    async fn list_repositories() -> Vec<RepositoryConfig>;
}

fn tracing_context() -> tarpc::context::Context {
    tarpc::context::current()
}

fn build_client_request(addr: &str, token: &str, request: ConnectHubRequest) -> Request<()> {
    Request::builder()
        .uri(format!("ws://{}/hub", addr))
        .header("Host", addr)
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header("Sec-WebSocket-Key", "unused")
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .header(
            CLIENT_REQUEST_HEADER.as_str(),
            serde_json::to_string(&request).unwrap(),
        )
        .body(())
        .unwrap()
}

#[derive(Clone)]
pub struct WorkerClient(HubClient);

pub async fn create_worker_client(
    addr: &str,
    token: &str,
    request: RegisterWorkerRequest,
) -> WorkerClient {
    let request = build_client_request(addr, token, ConnectHubRequest::Worker(request));
    let (socket, _) = connect_async(request).await.unwrap();
    WorkerClient(HubClient::new(Default::default(), WebSocketTransport::from(socket)).spawn())
}

impl EventLogger for WorkerClient {
    fn write(&self, x: LogEntry) {
        let context = tarpc::context::current();
        let client = self.0.clone();
        tokio::spawn(async move { client.write_log(context, x).await });
    }
}

#[async_trait]
impl CodeSearch for WorkerClient {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        match self
            .0
            .search(tracing_context(), q.to_owned(), limit, offset)
            .await
        {
            Ok(serp) => Ok(serp),
            Err(_) => Err(CodeSearchError::NotReady),
        }
    }

    async fn search_in_language(
        &self,
        git_url: &str,
        language: &str,
        tokens: &[String],
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        match self
            .0
            .search_in_language(
                tracing_context(),
                git_url.to_owned(),
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
pub enum ConnectHubRequest {
    Scheduler,
    Worker(RegisterWorkerRequest),
}

#[derive(Serialize, Deserialize)]
pub struct RegisterWorkerRequest {
    pub kind: WorkerKind,
    pub name: String,
    pub device: String,
    pub arch: String,
    pub cpu_info: String,
    pub cpu_count: i32,
    pub cuda_devices: Vec<String>,
    pub port: u16,
}

impl RegisterWorkerRequest {
    pub fn create_worker(self, addr: IpAddr) -> Worker {
        let port = self.port;
        let addr = format!("http://{addr}:{port}");
        Worker {
            name: self.name,
            kind: self.kind,
            addr,
            device: self.device,
            arch: self.arch,
            cpu_info: self.cpu_info,
            cpu_count: self.cpu_count,
            cuda_devices: self.cuda_devices,
        }
    }
}

pub static CLIENT_REQUEST_HEADER: HeaderName = HeaderName::from_static("x-tabby-client-request");

impl Header for ConnectHubRequest {
    fn name() -> &'static axum::http::HeaderName {
        &CLIENT_REQUEST_HEADER
    }

    fn decode<'i, I>(values: &mut I) -> Result<Self, axum_extra::headers::Error>
    where
        Self: Sized,
        I: Iterator<Item = &'i axum::http::HeaderValue>,
    {
        let mut x: Vec<_> = values
            .map(|x| serde_json::from_slice(x.as_bytes()))
            .collect();
        if let Some(x) = x.pop() {
            x.map_err(|_| axum_extra::headers::Error::invalid())
        } else {
            Err(axum_extra::headers::Error::invalid())
        }
    }

    fn encode<E: Extend<axum::http::HeaderValue>>(&self, _values: &mut E) {
        todo!()
    }
}

#[derive(Clone)]
pub struct SchedulerClient(HubClient);

pub async fn create_scheduler_client(addr: &str, token: &str) -> SchedulerClient {
    let request = build_client_request(addr, token, ConnectHubRequest::Scheduler);
    let (socket, _) = connect_async(request).await.unwrap();
    SchedulerClient(HubClient::new(Default::default(), WebSocketTransport::from(socket)).spawn())
}

#[async_trait]
impl RepositoryAccess for SchedulerClient {
    async fn list_repositories(&self) -> Result<Vec<RepositoryConfig>> {
        Ok(self.0.list_repositories(Context::current()).await?)
    }
}
