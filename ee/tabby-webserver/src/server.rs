mod proxy;
mod worker;

use std::net::SocketAddr;

use anyhow::Result;
use axum::{http::Request, middleware::Next, response::IntoResponse};
use hyper::{client::HttpConnector, Body, Client, StatusCode};
use tracing::{info, warn};

use crate::{
    api::{HubError, Worker, WorkerKind},
    db::DbConn,
};

pub struct ServerContext {
    client: Client<HttpConnector>,
    completion: worker::WorkerGroup,
    chat: worker::WorkerGroup,
    db_conn: DbConn,
}

impl ServerContext {
    pub fn new(db_conn: DbConn) -> Self {
        Self {
            client: Client::default(),
            completion: worker::WorkerGroup::default(),
            chat: worker::WorkerGroup::default(),
            db_conn,
        }
    }

    /// Query current token from the database.
    pub async fn read_registration_token(&self) -> Result<String> {
        self.db_conn.read_registration_token().await
    }

    /// Generate new token, and update it in the database.
    /// Return new token after update is done
    pub async fn reset_registration_token(&self) -> Result<String> {
        self.db_conn.reset_registration_token().await
    }

    pub async fn register_worker(&self, worker: Worker) -> Result<Worker, HubError> {
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
            Err(HubError::RequiresEnterpriseLicense)
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
