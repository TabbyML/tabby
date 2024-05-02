mod analytic;
mod auth;
pub mod background_job;
mod email;
pub mod event_logger;
mod job;
mod license;
mod proxy;
pub mod repository;
mod setting;
mod user_event;
mod worker;

use std::{net::SocketAddr, ops::Deref, sync::Arc};

use async_trait::async_trait;
use axum::{
    http::{HeaderName, HeaderValue, Request},
    middleware::Next,
    response::IntoResponse,
};
use hyper::{client::HttpConnector, Body, Client, StatusCode};
use juniper::ID;
use tabby_common::{
    api::{code::CodeSearch, event::EventLogger},
    constants::USER_HEADER_FIELD_NAME,
};
use tabby_db::DbConn;
use tabby_schema::{
    analytic::AnalyticService,
    auth::AuthenticationService,
    demo_mode,
    email::EmailService,
    job::JobService,
    license::{IsLicenseValid, LicenseService},
    repository::RepositoryService,
    setting::SettingService,
    user_event::UserEventService,
    worker::{RegisterWorkerError, Worker, WorkerKind, WorkerService},
    AsID, AsRowid, CoreError, Result, ServiceLocator,
};
use tracing::{info, warn};

use self::{
    analytic::new_analytic_service, email::new_email_service, license::new_license_service,
};
struct ServerContext {
    client: Client<HttpConnector>,
    completion: worker::WorkerGroup,
    chat: worker::WorkerGroup,
    db_conn: DbConn,
    mail: Arc<dyn EmailService>,
    auth: Arc<dyn AuthenticationService>,
    license: Arc<dyn LicenseService>,
    repository: Arc<dyn RepositoryService>,
    user_event: Arc<dyn UserEventService>,
    job: Arc<dyn JobService>,

    logger: Arc<dyn EventLogger>,
    code: Arc<dyn CodeSearch>,

    setting: Arc<dyn SettingService>,

    is_chat_enabled_locally: bool,
}

impl ServerContext {
    pub async fn new(
        logger: Arc<dyn EventLogger>,
        code: Arc<dyn CodeSearch>,
        repository: Arc<dyn RepositoryService>,
        db_conn: DbConn,
        is_chat_enabled_locally: bool,
    ) -> Self {
        let mail = Arc::new(
            new_email_service(db_conn.clone())
                .await
                .expect("failed to initialize mail service"),
        );
        let license = Arc::new(
            new_license_service(db_conn.clone())
                .await
                .expect("failed to initialize license service"),
        );
        let user_event = Arc::new(user_event::create(db_conn.clone()));
        let job = Arc::new(job::create(db_conn.clone()).await);
        let setting = Arc::new(setting::create(db_conn.clone()));
        Self {
            client: Client::default(),
            completion: worker::WorkerGroup::default(),
            chat: worker::WorkerGroup::default(),
            mail: mail.clone(),
            auth: Arc::new(auth::create(
                db_conn.clone(),
                mail,
                license.clone(),
                setting.clone(),
            )),
            license,
            repository,
            user_event,
            job,
            logger,
            code,
            setting,
            db_conn,
            is_chat_enabled_locally,
        }
    }

    /// Returns whether a request is authorized to access the content, and the user ID if authentication was used.
    async fn authorize_request(&self, request: &Request<Body>) -> (bool, Option<ID>) {
        let path = request.uri().path();
        if demo_mode()
            && (path.starts_with("/v1/completions")
                || path.starts_with("/v1/chat/completions")
                || path.starts_with("/v1beta/chat/completions"))
        {
            return (false, None);
        }
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

        // Allow JWT based access (from web browser), regardless of the license status.
        if let Ok(jwt) = self.auth.verify_access_token(token).await {
            return (true, Some(jwt.sub));
        }

        let is_license_valid = self.license.read().await.ensure_valid_license().is_ok();
        // If there's no valid license, only allows owner access.
        match self
            .db_conn
            .verify_auth_token(token, !is_license_valid)
            .await
        {
            Ok(id) => (true, Some(id.as_id())),
            Err(_) => (false, None),
        }
    }
}

#[async_trait]
impl WorkerService for ServerContext {
    /// Query current token from the database.
    async fn read_registration_token(&self) -> Result<String> {
        Ok(self.db_conn.read_registration_token().await?)
    }

    /// Generate new token, and update it in the database.
    /// Return new token after update is done
    async fn reset_registration_token(&self) -> Result<String> {
        Ok(self.db_conn.reset_registration_token().await?)
    }

    async fn list(&self) -> Vec<Worker> {
        [self.completion.list().await, self.chat.list().await].concat()
    }

    async fn register(&self, worker: Worker) -> Result<Worker, RegisterWorkerError> {
        let worker_group = match worker.kind {
            WorkerKind::Completion => &self.completion,
            WorkerKind::Chat => &self.chat,
        };

        let count_workers = worker_group.list().await.len();
        let license = self
            .license
            .read()
            .await
            .map_err(|_| RegisterWorkerError::RequiresTeamOrEnterpriseLicense)?;

        if !license.check_node_limit(count_workers + 1) {
            return Err(RegisterWorkerError::RequiresTeamOrEnterpriseLicense);
        }

        let worker = worker_group.register(worker).await;
        info!(
            "registering <{:?}> worker running at {}",
            worker.kind, worker.addr
        );
        Ok(worker)
    }

    async fn unregister(&self, worker_addr: &str) {
        let kind = if self.chat.unregister(worker_addr).await {
            WorkerKind::Chat
        } else if self.completion.unregister(worker_addr).await {
            WorkerKind::Completion
        } else {
            warn!(
                "Trying to unregister a worker missing in registry {}",
                worker_addr
            );
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
        } else if path.starts_with("/v1/chat/completions")
            || path.starts_with("/v1beta/chat/completions")
        {
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

    async fn is_chat_enabled(&self) -> Result<bool> {
        let num_chat_workers = self.chat.list().await.len();
        Ok(num_chat_workers > 0 || self.is_chat_enabled_locally)
    }
}

struct ArcServerContext(Arc<ServerContext>);

impl ArcServerContext {
    pub fn new(server_context: ServerContext) -> Self {
        Self(Arc::new(server_context))
    }
}

impl Deref for ArcServerContext {
    type Target = Arc<ServerContext>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ServiceLocator for ArcServerContext {
    fn auth(&self) -> Arc<dyn AuthenticationService> {
        self.auth.clone()
    }

    fn worker(&self) -> Arc<dyn WorkerService> {
        self.0.clone()
    }

    fn code(&self) -> Arc<dyn CodeSearch> {
        self.code.clone()
    }

    fn logger(&self) -> Arc<dyn EventLogger> {
        self.logger.clone()
    }

    fn job(&self) -> Arc<dyn JobService> {
        self.job.clone()
    }

    fn repository(&self) -> Arc<dyn RepositoryService> {
        self.repository.clone()
    }

    fn email(&self) -> Arc<dyn EmailService> {
        self.mail.clone()
    }

    fn setting(&self) -> Arc<dyn SettingService> {
        self.setting.clone()
    }

    fn license(&self) -> Arc<dyn LicenseService> {
        self.license.clone()
    }

    fn analytic(&self) -> Arc<dyn AnalyticService> {
        new_analytic_service(self.db_conn.clone())
    }

    fn user_event(&self) -> Arc<dyn UserEventService> {
        self.user_event.clone()
    }
}

pub async fn create_service_locator(
    logger: Arc<dyn EventLogger>,
    code: Arc<dyn CodeSearch>,
    repository: Arc<dyn RepositoryService>,
    db: DbConn,
    is_chat_enabled: bool,
) -> Arc<dyn ServiceLocator> {
    Arc::new(ArcServerContext::new(
        ServerContext::new(logger, code, repository, db, is_chat_enabled).await,
    ))
}

/// Returns (limit, skip_id, backwards)
pub fn graphql_pagination_to_filter(
    after: Option<String>,
    before: Option<String>,
    first: Option<usize>,
    last: Option<usize>,
) -> Result<(Option<usize>, Option<i32>, bool), CoreError> {
    match (first, last) {
        (Some(first), None) => {
            let after = after
                .map(|x| ID::new(x).as_rowid())
                .transpose()?
                .map(|x| x as i32);
            Ok((Some(first), after, false))
        }
        (None, Some(last)) => {
            let before = before
                .map(|x| ID::new(x).as_rowid())
                .transpose()?
                .map(|x| x as i32);
            Ok((Some(last), before, true))
        }
        _ => Ok((None, None, false)),
    }
}
