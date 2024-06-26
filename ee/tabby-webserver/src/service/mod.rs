mod analytic;
mod auth;
pub mod background_job;
mod email;
pub mod event_logger;
pub mod integration;
pub mod job;
mod license;
pub mod repository;
mod setting;
mod user_event;
pub mod web_crawler;

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{HeaderName, HeaderValue, Request, StatusCode},
    middleware::Next,
    response::IntoResponse,
};
use hyper::{HeaderMap, Uri};
use juniper::ID;
use tabby_common::{
    api::{code::CodeSearch, event::EventLogger},
    constants::USER_HEADER_FIELD_NAME,
};
use tabby_db::DbConn;
use tabby_schema::{
    analytic::AnalyticService,
    auth::AuthenticationService,
    email::EmailService,
    integration::IntegrationService,
    is_demo_mode,
    job::JobService,
    license::{IsLicenseValid, LicenseService},
    repository::RepositoryService,
    setting::SettingService,
    user_event::UserEventService,
    web_crawler::WebCrawlerService,
    worker::WorkerService,
    AsID, AsRowid, CoreError, Result, ServiceLocator,
};

use self::{
    analytic::new_analytic_service, email::new_email_service, license::new_license_service,
};
struct ServerContext {
    db_conn: DbConn,
    mail: Arc<dyn EmailService>,
    auth: Arc<dyn AuthenticationService>,
    license: Arc<dyn LicenseService>,
    repository: Arc<dyn RepositoryService>,
    integration: Arc<dyn IntegrationService>,
    user_event: Arc<dyn UserEventService>,
    job: Arc<dyn JobService>,
    web_crawler: Arc<dyn WebCrawlerService>,

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
        integration: Arc<dyn IntegrationService>,
        web_crawler: Arc<dyn WebCrawlerService>,
        job: Arc<dyn JobService>,
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
        let setting = Arc::new(setting::create(db_conn.clone()));

        Self {
            mail: mail.clone(),
            auth: Arc::new(auth::create(
                db_conn.clone(),
                mail,
                license.clone(),
                setting.clone(),
            )),
            web_crawler,
            license,
            repository,
            integration,
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
    async fn authorize_request(&self, uri: &Uri, headers: &HeaderMap) -> (bool, Option<ID>) {
        let path = uri.path();
        if !(path.starts_with("/v1/") || path.starts_with("/v1beta/")) {
            return (true, None);
        }
        let authorization = headers
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
        let requires_owner = !is_license_valid || is_demo_mode();

        // If there's no valid license, only allows owner access.
        match self.db_conn.verify_auth_token(token, requires_owner).await {
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

    async fn dispatch_request(
        &self,
        mut request: Request<Body>,
        next: Next,
    ) -> axum::response::Response {
        let (auth, user) = self
            .authorize_request(request.uri(), request.headers())
            .await;
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

        next.run(request).await
    }

    async fn is_chat_enabled(&self) -> Result<bool> {
        Ok(self.is_chat_enabled_locally)
    }
}

struct ArcServerContext(Arc<ServerContext>);

impl ArcServerContext {
    pub fn new(server_context: ServerContext) -> Self {
        Self(Arc::new(server_context))
    }
}

impl ServiceLocator for ArcServerContext {
    fn auth(&self) -> Arc<dyn AuthenticationService> {
        self.0.auth.clone()
    }

    fn worker(&self) -> Arc<dyn WorkerService> {
        self.0.clone()
    }

    fn code(&self) -> Arc<dyn CodeSearch> {
        self.0.code.clone()
    }

    fn logger(&self) -> Arc<dyn EventLogger> {
        self.0.logger.clone()
    }

    fn job(&self) -> Arc<dyn JobService> {
        self.0.job.clone()
    }

    fn repository(&self) -> Arc<dyn RepositoryService> {
        self.0.repository.clone()
    }

    fn email(&self) -> Arc<dyn EmailService> {
        self.0.mail.clone()
    }

    fn setting(&self) -> Arc<dyn SettingService> {
        self.0.setting.clone()
    }

    fn license(&self) -> Arc<dyn LicenseService> {
        self.0.license.clone()
    }

    fn analytic(&self) -> Arc<dyn AnalyticService> {
        new_analytic_service(self.0.db_conn.clone())
    }

    fn user_event(&self) -> Arc<dyn UserEventService> {
        self.0.user_event.clone()
    }

    fn integration(&self) -> Arc<dyn IntegrationService> {
        self.0.integration.clone()
    }

    fn web_crawler(&self) -> Arc<dyn WebCrawlerService> {
        self.0.web_crawler.clone()
    }
}

pub async fn create_service_locator(
    logger: Arc<dyn EventLogger>,
    code: Arc<dyn CodeSearch>,
    repository: Arc<dyn RepositoryService>,
    integration: Arc<dyn IntegrationService>,
    web_crawler: Arc<dyn WebCrawlerService>,
    job: Arc<dyn JobService>,
    db: DbConn,
    is_chat_enabled: bool,
) -> Arc<dyn ServiceLocator> {
    Arc::new(ArcServerContext::new(
        ServerContext::new(
            logger,
            code,
            repository,
            integration,
            web_crawler,
            job,
            db,
            is_chat_enabled,
        )
        .await,
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

pub async fn create_gitlab_client(
    api_base: &str,
    access_token: &str,
) -> Result<gitlab::AsyncGitlab, anyhow::Error> {
    let url = url::Url::parse(api_base)?;
    let api_base = url.authority();
    let mut builder = gitlab::Gitlab::builder(api_base.to_owned(), access_token);
    if url.scheme() == "http" {
        builder.insecure();
    };
    Ok(builder.build_async().await?)
}
