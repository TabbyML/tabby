mod auth;
mod cron;
mod db;
mod proxy;
mod worker;

use std::{net::SocketAddr, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use axum::{
    http::{HeaderValue, Request},
    middleware::Next,
    response::IntoResponse,
};
use hyper::{client::HttpConnector, Body, Client, StatusCode};
use tabby_common::api::{code::CodeSearch, event::RawEventLogger};
use tracing::{info, warn};

use self::{cron::run_cron, db::DbConn};
use crate::schema::{
    auth::AuthenticationService,
    worker::{RegisterWorkerError, Worker, WorkerKind, WorkerService},
    ServiceLocator,
};

struct ServerContext {
    client: Client<HttpConnector>,
    completion: worker::WorkerGroup,
    chat: worker::WorkerGroup,
    db_conn: DbConn,

    logger: Arc<dyn RawEventLogger>,
    code: Arc<dyn CodeSearch>,
}

impl ServerContext {
    pub async fn new(logger: Arc<dyn RawEventLogger>, code: Arc<dyn CodeSearch>) -> Self {
        let db_conn = DbConn::new().await.unwrap();
        run_cron(&db_conn);
        Self {
            client: Client::default(),
            completion: worker::WorkerGroup::default(),
            chat: worker::WorkerGroup::default(),
            db_conn,
            logger,
            code,
        }
    }

    async fn authorize_request(&self, request: &Request<Body>) -> bool {
        let path = request.uri().path();
        if path.starts_with("/v1/") || path.starts_with("/v1beta/") {
            let token = {
                let authorization = request
                    .headers()
                    .get("authorization")
                    .map(HeaderValue::to_str)
                    .and_then(Result::ok);

                if let Some(authorization) = authorization {
                    let split = authorization.split_once(' ');
                    match split {
                        // Found proper bearer
                        Some(("Bearer", contents)) => Some(contents),
                        _ => None,
                    }
                } else {
                    None
                }
            };

            if let Some(token) = token {
                if self.db_conn.verify_access_token(token).await.is_err()
                    && !self.db_conn.verify_auth_token(token).await
                {
                    return false;
                }
            } else {
                // Admin system is initialized, but there's no valid token.
                return false;
            }
        }

        true
    }
}

#[async_trait]
impl WorkerService for ServerContext {
    /// Query current token from the database.
    async fn read_registration_token(&self) -> Result<String> {
        self.db_conn.read_registration_token().await
    }

    /// Generate new token, and update it in the database.
    /// Return new token after update is done
    async fn reset_registration_token(&self) -> Result<String> {
        self.db_conn.reset_registration_token().await
    }

    async fn list_workers(&self) -> Vec<Worker> {
        [self.completion.list().await, self.chat.list().await].concat()
    }

    async fn register_worker(&self, worker: Worker) -> Result<Worker, RegisterWorkerError> {
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
            Err(RegisterWorkerError::RequiresEnterpriseLicense)
        }
    }

    async fn unregister_worker(&self, worker_addr: &str) {
        let kind = if self.chat.unregister(worker_addr).await {
            WorkerKind::Chat
        } else if self.completion.unregister(worker_addr).await {
            WorkerKind::Completion
        } else {
            warn!("Trying to unregister a worker missing in registry");
            return;
        };

        info!("unregistering <{:?}> worker at {}", kind, worker_addr);
    }

    async fn dispatch_request(
        &self,
        request: Request<Body>,
        next: Next<Body>,
    ) -> axum::response::Response {
        if !self.authorize_request(&request).await {
            return axum::response::Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .body(Body::empty())
                .unwrap()
                .into_response();
        }

        let remote_addr = request
            .extensions()
            .get::<axum::extract::ConnectInfo<SocketAddr>>()
            .map(|ci| ci.0)
            .expect("Unable to extract remote addr");

        let path = request.uri().path();
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

impl ServiceLocator for ServerContext {
    fn auth(&self) -> &dyn AuthenticationService {
        &self.db_conn
    }

    fn worker(&self) -> &dyn WorkerService {
        self
    }

    fn code(&self) -> &dyn CodeSearch {
        &*self.code
    }

    fn logger(&self) -> &dyn RawEventLogger {
        &*self.logger
    }
}

pub async fn create_service_locator(
    logger: Arc<dyn RawEventLogger>,
    code: Arc<dyn CodeSearch>,
) -> Arc<dyn ServiceLocator> {
    Arc::new(ServerContext::new(logger, code).await)
}
