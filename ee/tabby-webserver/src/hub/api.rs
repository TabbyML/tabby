use std::net::IpAddr;

use anyhow::Result;
use async_trait::async_trait;
use axum::{headers::Header, http::HeaderName};
use hyper::Request;
use serde::{Deserialize, Serialize};
use tabby_common::{
    api::{
        code::{CodeSearch, CodeSearchError, SearchResponse},
        event::RawEventLogger,
    },
    config::{RepositoryAccess, RepositoryConfig},
};
use tarpc::context::Context;
use tokio_tungstenite::connect_async;

use super::websocket::WebSocketTransport;
use crate::schema::worker::Worker;
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

    async fn list_repositories() -> Vec<RepositoryConfig>;
    async fn create_job_run(name: String) -> i32;
    async fn update_job_output(id: i32, stdout: String, stderr: String);
    async fn complete_job_run(id: i32, exit_code: i32);
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
            &CLIENT_REQUEST_HEADER,
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

impl RawEventLogger for WorkerClient {
    fn log(&self, content: String) {
        let context = tarpc::context::current();
        let client = self.0.clone();

        tokio::spawn(async move { client.log_event(context, content).await });
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
        language: &str,
        tokens: &[String],
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        match self
            .0
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
    pub fn into_worker(self, addr: IpAddr) -> Worker {
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
    async fn create_job_run(&self, name: String) -> Result<i32> {
        Ok(self.0.create_job_run(tracing_context(), name).await?)
    }
    async fn update_job_output(&self, id: i32, stdout: String, stderr: String) -> Result<()> {
        Ok(self
            .0
            .update_job_output(tracing_context(), id, stdout, stderr)
            .await?)
    }
    async fn complete_job_run(&self, id: i32, exit_code: i32) -> Result<()> {
        Ok(self
            .0
            .complete_job_run(tracing_context(), id, exit_code)
            .await?)
    }
}
