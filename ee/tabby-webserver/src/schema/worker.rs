use anyhow::Result;
use async_trait::async_trait;
use axum::middleware::Next;
use hyper::{Body, Request};
use juniper::{GraphQLEnum, GraphQLObject};
use serde::{Deserialize, Serialize};
use thiserror::Error;

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
pub enum RegisterWorkerError {
    #[error("Invalid token")]
    InvalidToken(String),

    #[error("Feature requires enterprise license")]
    RequiresEnterpriseLicense,

    #[error("Each hub client should only calls register_worker once")]
    RegisterWorkerOnce,
}

#[async_trait]
pub trait WorkerService: Send + Sync {
    async fn read_registration_token(&self) -> Result<String>;
    async fn reset_registration_token(&self) -> Result<String>;
    async fn list_workers(&self) -> Vec<Worker>;
    async fn register_worker(&self, worker: Worker) -> Result<Worker, RegisterWorkerError>;
    async fn unregister_worker(&self, worker_addr: &str);
    async fn dispatch_request(
        &self,
        request: Request<Body>,
        next: Next<Body>,
    ) -> axum::response::Response;
}
