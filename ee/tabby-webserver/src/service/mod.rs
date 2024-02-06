mod auth;
mod cron;
mod email;
mod job;
mod proxy;
mod repository;
mod worker;

use std::{net::SocketAddr, num::ParseIntError, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use axum::{
    http::{HeaderName, HeaderValue, Request},
    middleware::Next,
    response::IntoResponse,
};
use hyper::{client::HttpConnector, Body, Client, StatusCode};
use tabby_common::{
    api::{code::CodeSearch, event::RawEventLogger},
    constants::USER_HEADER_FIELD_NAME,
};
use tabby_db::DbConn;
use tracing::{info, warn};

use self::{cron::run_cron, email::new_email_service};
use crate::schema::{
    auth::AuthenticationService,
    email::EmailService,
    job::JobService,
    repository::RepositoryService,
    worker::{RegisterWorkerError, Worker, WorkerKind, WorkerService},
    ServiceLocator,
};

struct ServerContext {
    client: Client<HttpConnector>,
    completion: worker::WorkerGroup,
    chat: worker::WorkerGroup,
    db_conn: DbConn,
    mail_service: Arc<dyn EmailService>,

    logger: Arc<dyn RawEventLogger>,
    code: Arc<dyn CodeSearch>,
}

impl ServerContext {
    pub async fn new(
        logger: Arc<dyn RawEventLogger>,
        code: Arc<dyn CodeSearch>,
        local_listen_port: u16,
    ) -> Self {
        let db_conn = DbConn::new().await.unwrap();
        run_cron(
            &db_conn,
            format!("http://localhost:{}", local_listen_port),
            false,
        )
        .await;
        Self {
            client: Client::default(),
            completion: worker::WorkerGroup::default(),
            chat: worker::WorkerGroup::default(),
            mail_service: Arc::new(
                new_email_service(db_conn.clone())
                    .await
                    .expect("failed to initialize mail service"),
            ),
            db_conn,
            logger,
            code,
        }
    }

    async fn authorize_request(&self, request: &Request<Body>) -> (bool, Option<String>) {
        let path = request.uri().path();
        if !(path.starts_with("/v1/") || path.starts_with("/v1beta/")) {
            return (true, None);
        }
        let authorization = request
            .headers()
            .get("authorization")
            .map(HeaderValue::to_str)
            .and_then(Result::ok);

        let token = authorization
            .and_then(|s| s.split_once(' '))
            .map(|(_bearer, token)| token);

        let Some(token) = token else {
            // Admin system is initialized, but there is no valid token.
            return (false, None);
        };
        if let Ok(jwt) = self.db_conn.verify_access_token(token).await {
            return (true, Some(jwt.claims.sub));
        }
        match self.db_conn.verify_auth_token(token).await {
            Ok(email) => (true, Some(email)),
            Err(_) => (false, None),
        }
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
        mut request: Request<Body>,
        next: Next<Body>,
    ) -> axum::response::Response {
        let (auth, user) = self.authorize_request(&request).await;
        if !auth {
            return axum::response::Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .body(Body::empty())
                .unwrap()
                .into_response();
        }

        if let Some(user) = user {
            request.headers_mut().append(
                HeaderName::from_static(USER_HEADER_FIELD_NAME),
                HeaderValue::from_str(&user).expect("User must be valid header"),
            );
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

impl ServiceLocator for Arc<ServerContext> {
    fn auth(&self) -> Arc<dyn AuthenticationService> {
        Arc::new(self.db_conn.clone())
    }

    fn worker(&self) -> Arc<dyn WorkerService> {
        self.clone()
    }

    fn code(&self) -> Arc<dyn CodeSearch> {
        self.code.clone()
    }

    fn logger(&self) -> Arc<dyn RawEventLogger> {
        self.logger.clone()
    }

    fn job(&self) -> Arc<dyn JobService> {
        Arc::new(self.db_conn.clone())
    }

    fn repository(&self) -> Arc<dyn RepositoryService> {
        Arc::new(self.db_conn.clone())
    }

    fn email(&self) -> Arc<dyn EmailService> {
        self.mail_service.clone()
    }
}

pub async fn create_service_locator(
    logger: Arc<dyn RawEventLogger>,
    code: Arc<dyn CodeSearch>,
    local_listen_port: u16,
) -> Arc<dyn ServiceLocator> {
    Arc::new(Arc::new(
        ServerContext::new(logger, code, local_listen_port).await,
    ))
}

pub fn graphql_pagination_to_filter(
    after: Option<String>,
    before: Option<String>,
    first: Option<usize>,
    last: Option<usize>,
) -> Result<(Option<usize>, Option<i32>, bool), ParseIntError> {
    match (first, last) {
        (Some(first), None) => {
            let after = after.map(|x| x.parse::<i32>()).transpose()?;
            Ok((Some(first), after, false))
        }
        (None, Some(last)) => {
            let before = before.map(|x| x.parse::<i32>()).transpose()?;
            Ok((Some(last), before, true))
        }
        _ => Ok((None, None, false)),
    }
}
