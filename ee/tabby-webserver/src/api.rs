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
pub enum WebserverApiError {
    #[error("Invalid worker token")]
    InvalidToken(String),

    #[error("Feature requires enterprise license")]
    RequiresEnterpriseLicense,
}

#[tarpc::service]
pub trait WebserverApi {
    async fn register_worker(
        kind: WorkerKind,
        port: i32,
        name: String,
        device: String,
        arch: String,
        cpu_info: String,
        cpu_count: i32,
        cuda_devices: Vec<String>,
    ) -> Result<Worker, WebserverApiError>;
}

pub fn tracing_context() -> tarpc::context::Context {
    tarpc::context::current()
}

pub async fn create_client(addr: String) -> WebserverApiClient {
    let (socket, _) = connect_async(&addr).await.unwrap();
    WebserverApiClient::new(Default::default(), WebSocketTransport::from(socket)).spawn()
}
