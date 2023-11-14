mod proxy;

use std::{net::SocketAddr, sync::Arc};

use axum::{http::Request, middleware::Next, response::IntoResponse};
use hyper::{client::HttpConnector, Body, Client, StatusCode};
use thiserror::Error;
use tracing::{info, warn};

use crate::{
    api::{WebserverApi, Worker, WorkerKind},
    worker,
};

#[derive(Error, Debug)]
pub enum WebserverError {
    #[error("Invalid worker token")]
    InvalidToken(String),

    #[error("Feature requires enterprise license")]
    RequiresEnterpriseLicense,
}

#[derive(Default)]
pub struct Webserver {
    client: Client<HttpConnector>,
    completion: worker::WorkerGroup,
    chat: worker::WorkerGroup,
}

impl Webserver {
    pub async fn register_worker(&self, worker: Worker) -> Result<Worker, WebserverError> {
        let worker = match worker.kind {
            WorkerKind::Completion => self.completion.register(worker).await,
            WorkerKind::Chat => self.chat.register(worker).await,
        };

        if let Some(worker) = worker {
            info!(
                "registering <{:?}> worker running at {}",
                worker.kind, worker.addr
            );
            Ok(worker)
        } else {
            Err(WebserverError::RequiresEnterpriseLicense)
        }
    }

    pub async fn list_workers(&self) -> Vec<Worker> {
        [self.completion.list().await, self.chat.list().await].concat()
    }

    pub async fn dispatch_request(
        &self,
        request: Request<Body>,
        next: Next<Body>,
    ) -> axum::response::Response {
        let path = request.uri().path();

        let remote_addr = request
            .extensions()
            .get::<axum::extract::ConnectInfo<SocketAddr>>()
            .map(|ci| ci.0)
            .expect("Unable to extract remote addr");

        let worker = if path.starts_with("/v1/completions") {
            self.completion.select().await
        } else if path.starts_with("/v1beta/chat/completions") {
            self.chat.select().await
        } else {
            None
        };

        if let Some(worker) = worker {
            match proxy::call(self.client.clone(), remote_addr.ip(), &worker, request).await {
                Ok(res) => res.into_response(),
                Err(err) => {
                    warn!("Failed to proxy request {}", err);
                    axum::response::Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::empty())
                        .unwrap()
                        .into_response()
                }
            }
        } else {
            next.run(request).await
        }
    }
}

pub struct WebserverImpl {
    ws: Arc<Webserver>,
    conn: SocketAddr,
}

impl WebserverImpl {
    pub fn new(ws: Arc<Webserver>, conn: SocketAddr) -> Self {
        Self { ws, conn }
    }
}

#[tarpc::server]
impl WebserverApi for Arc<WebserverImpl> {
    async fn register_worker_as(
        self,
        _context: tarpc::context::Context,
        kind: WorkerKind,
        port: i32,
        name: String,
        device: String,
        arch: String,
        cpu_info: String,
        cpu_count: i32,
        cuda_devices: Vec<String>,
    ) -> Worker {
        let worker = Worker {
            name,
            kind,
            addr: format!("http://{}:{}", self.conn.ip(), port),
            device,
            arch,
            cpu_info,
            cpu_count,
            cuda_devices,
        };
        self.ws.register_worker(worker).await.unwrap()
    }
}
