use juniper::{GraphQLEnum, GraphQLObject};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio_tungstenite::connect_async;

use crate::websocket::WebSocketTransport;

#[derive(GraphQLEnum, Serialize, Deserialize, Clone, Debug)]
pub enum WorkerKind {
    Completion,
    Chat,
}

#[derive(GraphQLObject, Serialize, Deserialize, Clone, Debug)]
pub struct Worker {
    pub kind: WorkerKind,
    pub name: String,
    pub addr: String,
    pub device: String,
    pub arch: String,
    pub cpu_info: String,
    pub cpu_count: i32,
    pub cuda_devices: Vec<String>,
}

#[derive(Serialize, Deserialize, Error, Debug)]
pub enum HubError {
    #[error("Invalid token")]
    InvalidToken(String),

    #[error("Feature requires enterprise license")]
    RequiresEnterpriseLicense,
}

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
        cuda_devices: Vec<String>,
        token: String,
    ) -> Result<Worker, HubError>;

    async fn log_event(content: String);
}

pub fn tracing_context() -> tarpc::context::Context {
    tarpc::context::current()
}

pub async fn create_client(addr: &str) -> HubClient {
    let addr = format!("ws://{}/hub", addr);
    let (socket, _) = connect_async(&addr).await.unwrap();
    HubClient::new(Default::default(), WebSocketTransport::from(socket)).spawn()
}
