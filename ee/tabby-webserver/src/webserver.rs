mod proxy;

use std::net::SocketAddr;

use axum::{http::Request, middleware::Next, response::IntoResponse};
use hyper::{client::HttpConnector, Body, Client, StatusCode};
use thiserror::Error;
use tracing::{info, warn};

use crate::{
    schema::{Worker, WorkerKind},
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

// FIXME: generate token and support refreshing in database.
static WORKER_TOKEN: &str = "4c749fad-2be7-45a3-849e-7714ccade382";

impl Webserver {
    pub async fn register_worker(
        &self,
        token: String,
        client_addr: SocketAddr,
        kind: WorkerKind,
        port: i32,
    ) -> Result<Worker, WebserverError> {
        if token != WORKER_TOKEN {
            return Err(WebserverError::InvalidToken(token));
        }

        let addr = SocketAddr::new(client_addr.ip(), port as u16);
        let addr = match kind {
            WorkerKind::Completion => self.completion.register(addr).await,
            WorkerKind::Chat => self.chat.register(addr).await,
        };

        if let Some(addr) = addr {
            info!("registering <{:?}> worker running at {}", kind, addr);
            Ok(Worker::new(kind, addr))
        } else {
            Err(WebserverError::RequiresEnterpriseLicense)
        }
    }

    pub async fn list_workers(&self) -> Vec<Worker> {
        let make_workers = |x: WorkerKind, lst: Vec<String>| -> Vec<Worker> {
            lst.into_iter()
                .map(|addr| Worker::new(x.clone(), addr))
                .collect()
        };

        [
            make_workers(WorkerKind::Completion, self.completion.list().await),
            make_workers(WorkerKind::Chat, self.chat.list().await),
        ]
        .concat()
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
