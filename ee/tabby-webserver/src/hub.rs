use anyhow::Result;
use axum::{extract::Request, http::HeaderName};
use axum_extra::headers::Header;
use serde::{Deserialize, Serialize};
use tabby_common::api::event::{EventLogger, LogEntry};
use tokio_tungstenite::connect_async;

use crate::axum::websocket::WebSocketTransport;

#[tarpc::service]
pub trait Hub {
    async fn write_log(x: LogEntry);
}

fn tracing_context() -> tarpc::context::Context {
    tarpc::context::current()
}

fn build_client_request(addr: &str, token: &str, request: ConnectHubRequest) -> Request<()> {
    Request::builder()
        .uri(format!("ws://{addr}/hub"))
        .header("Host", addr)
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header("Sec-WebSocket-Key", "unused")
        .header("Authorization", format!("Bearer {token}"))
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

pub async fn create_worker_client(addr: &str, token: &str) -> WorkerClient {
    let request = build_client_request(addr, token, ConnectHubRequest);
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

#[derive(Serialize, Deserialize)]
pub struct ConnectHubRequest;

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
