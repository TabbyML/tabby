mod access_policy;
mod analytic;
pub mod answer;
mod auth;
pub mod background_job;
pub mod context;
mod email;
pub mod embedding;
pub mod event_logger;
pub mod integration;
pub mod job;
mod license;
mod notification;
mod page;
mod preset_web_documents_data;
pub mod repository;
mod setting;
mod thread;
mod user_event;
mod user_group;
pub mod utils;
pub mod web_documents;

use std::sync::Arc;

use answer::AnswerService;
use anyhow::Context;
use async_trait::async_trait;
pub use auth::create as new_auth_service;
#[cfg(test)]
pub use auth::testutils::FakeAuthService;
use axum::{
    body::Body,
    http::{HeaderName, HeaderValue, Request, StatusCode},
    middleware::Next,
    response::IntoResponse,
};
pub use email::new_email_service;
use hyper::{HeaderMap, Uri};
use juniper::ID;
pub use license::new_license_service;
pub use setting::create as new_setting_service;
use tabby_common::{
    api::{code::CodeSearch, event::EventLogger},
    constants::USER_HEADER_FIELD_NAME,
};
use tabby_db::{DbConn, UserDAO, UserGroupDAO};
use tabby_inference::{ChatCompletionStream, CompletionStream, Embedding as EmbeddingService};
use tabby_schema::{
    access_policy::AccessPolicyService,
    analytic::AnalyticService,
    auth::{AuthenticationService, UserSecured},
    context::ContextService,
    email::EmailService,
    integration::IntegrationService,
    interface::UserValue,
    is_demo_mode,
    job::JobService,
    license::{IsLicenseValid, LicenseService},
    notification::NotificationService,
    page::PageService,
    policy,
    repository::RepositoryService,
    setting::SettingService,
    thread::ThreadService,
    user_event::UserEventService,
    user_group::{UserGroup, UserGroupMembership, UserGroupService},
    web_documents::WebDocumentService,
    worker::WorkerService,
    AsID, AsRowid, CoreError, Result, ServiceLocator,
};

use self::analytic::new_analytic_service;
use crate::rate_limit::UserRateLimiter;

struct ServerContext {
    db_conn: DbConn,
    mail: Arc<dyn EmailService>,
    embedding: Arc<dyn EmbeddingService>,
    chat: Option<Arc<dyn ChatCompletionStream>>,
    completion: Option<Arc<dyn CompletionStream>>,
    auth: Arc<dyn AuthenticationService>,
    notification: Arc<dyn NotificationService>,
    license: Arc<dyn LicenseService>,
    repository: Arc<dyn RepositoryService>,
    integration: Arc<dyn IntegrationService>,
    user_event: Arc<dyn UserEventService>,
    job: Arc<dyn JobService>,
    web_documents: Arc<dyn WebDocumentService>,
    thread: Arc<dyn ThreadService>,
    page: Option<Arc<dyn PageService>>,
    context: Arc<dyn ContextService>,
    user_group: Arc<dyn UserGroupService>,
    access_policy: Arc<dyn AccessPolicyService>,

    logger: Arc<dyn EventLogger>,
    code: Arc<dyn CodeSearch>,

    setting: Arc<dyn SettingService>,

    user_rate_limiter: UserRateLimiter,
}

impl ServerContext {
    pub async fn new(
        logger: Arc<dyn EventLogger>,
        auth: Arc<dyn AuthenticationService>,
        chat: Option<Arc<dyn ChatCompletionStream>>,
        completion: Option<Arc<dyn CompletionStream>>,
        code: Arc<dyn CodeSearch>,
        repository: Arc<dyn RepositoryService>,
        integration: Arc<dyn IntegrationService>,
        job: Arc<dyn JobService>,
        answer: Option<Arc<AnswerService>>,
        context: Arc<dyn ContextService>,
        web_documents: Arc<dyn WebDocumentService>,
        mail: Arc<dyn EmailService>,
        license: Arc<dyn LicenseService>,
        setting: Arc<dyn SettingService>,
        db_conn: DbConn,
        embedding: Arc<dyn EmbeddingService>,
    ) -> Self {
        let user_event = Arc::new(user_event::create(db_conn.clone()));
        let thread = Arc::new(thread::create(
            db_conn.clone(),
            answer.clone(),
            Some(auth.clone()),
            context.clone(),
        ));
        let page = chat.as_ref().map(|chat| {
            Arc::new(page::create(
                db_conn.clone(),
                chat.clone(),
                thread.clone(),
                context.clone(),
            )) as Arc<dyn PageService>
        });

        let user_group = Arc::new(user_group::create(db_conn.clone()));
        let access_policy = Arc::new(access_policy::create(db_conn.clone(), context.clone()));
        let notification = Arc::new(notification::create(db_conn.clone()));

        background_job::start(
            db_conn.clone(),
            job.clone(),
            repository.git(),
            repository.third_party(),
            integration.clone(),
            repository.clone(),
            context.clone(),
            license.clone(),
            notification.clone(),
            embedding.clone(),
        )
        .await;

        Self {
            mail,
            embedding,
            chat,
            completion,
            auth,
            web_documents,
            thread,
            page,
            context,
            license,
            repository,
            integration,
            user_event,
            job,
            logger,
            code,
            setting,
            user_group,
            access_policy,
            notification,
            db_conn,
            user_rate_limiter: UserRateLimiter::default(),
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
        let unauthorized = axum::response::Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .body(Body::empty())
            .unwrap()
            .into_response();
        if !auth {
            return unauthorized;
        }

        if let Some(user) = user {
            // Apply rate limiting when `user` is not none.
            if !self
                .user_rate_limiter
                .is_allowed(request.uri(), &user)
                .await
            {
                return axum::response::Response::builder()
                    .status(StatusCode::TOO_MANY_REQUESTS)
                    .body(Body::empty())
                    .unwrap()
                    .into_response();
            }

            request.headers_mut().append(
                HeaderName::from_static(USER_HEADER_FIELD_NAME),
                HeaderValue::from_str(&user).expect("User must be valid header"),
            );

            let Ok(user) = self.auth.get_user(&user).await else {
                return unauthorized;
            };

            let Ok(context_info) = self.context.read(Some(&user.policy)).await else {
                return unauthorized;
            };

            request
                .extensions_mut()
                .insert(context_info.allowed_code_repository());
        }

        next.run(request).await
    }

    async fn is_chat_enabled(&self) -> Result<bool> {
        Ok(self.chat.is_some())
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

    fn chat(&self) -> Option<Arc<dyn ChatCompletionStream>> {
        self.0.chat.clone()
    }

    fn worker(&self) -> Arc<dyn WorkerService> {
        self.0.clone()
    }

    fn code(&self) -> Arc<dyn CodeSearch> {
        self.0.code.clone()
    }

    fn completion(&self) -> Option<Arc<dyn CompletionStream>> {
        self.0.completion.clone()
    }

    fn logger(&self) -> Arc<dyn EventLogger> {
        self.0.logger.clone()
    }

    fn notification(&self) -> Arc<dyn tabby_schema::notification::NotificationService> {
        self.0.notification.clone()
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

    fn embedding(&self) -> Arc<dyn EmbeddingService> {
        self.0.embedding.clone()
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

    fn web_documents(&self) -> Arc<dyn WebDocumentService> {
        self.0.web_documents.clone()
    }

    fn thread(&self) -> Arc<dyn ThreadService> {
        self.0.thread.clone()
    }

    fn page(&self) -> Option<Arc<dyn PageService>> {
        self.0.page.clone()
    }

    fn context(&self) -> Arc<dyn ContextService> {
        self.0.context.clone()
    }

    fn user_group(&self) -> Arc<dyn UserGroupService> {
        self.0.user_group.clone()
    }

    fn access_policy(&self) -> Arc<dyn AccessPolicyService> {
        self.0.access_policy.clone()
    }
}

pub async fn create_service_locator(
    logger: Arc<dyn EventLogger>,
    auth: Arc<dyn AuthenticationService>,
    chat: Option<Arc<dyn ChatCompletionStream>>,
    completion: Option<Arc<dyn CompletionStream>>,
    code: Arc<dyn CodeSearch>,
    repository: Arc<dyn RepositoryService>,
    integration: Arc<dyn IntegrationService>,
    job: Arc<dyn JobService>,
    answer: Option<Arc<AnswerService>>,
    context: Arc<dyn ContextService>,
    web_documents: Arc<dyn WebDocumentService>,
    mail: Arc<dyn EmailService>,
    license: Arc<dyn LicenseService>,
    setting: Arc<dyn SettingService>,
    db: DbConn,
    embedding: Arc<dyn EmbeddingService>,
) -> Arc<dyn ServiceLocator> {
    Arc::new(ArcServerContext::new(
        ServerContext::new(
            logger,
            auth,
            chat,
            completion,
            code,
            repository,
            integration,
            job,
            answer,
            context,
            web_documents,
            mail,
            license,
            setting,
            db,
            embedding,
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

trait UserSecuredExt {
    fn new(db: DbConn, val: UserDAO) -> tabby_schema::auth::UserSecured;
}

impl UserSecuredExt for tabby_schema::auth::UserSecured {
    fn new(db: DbConn, val: UserDAO) -> tabby_schema::auth::UserSecured {
        let is_owner = val.is_owner();
        let id = val.id.as_id();
        tabby_schema::auth::UserSecured {
            policy: policy::AccessPolicy::new(db, &id, val.is_admin),
            id,
            email: val.email,
            name: val.name.unwrap_or_default(),
            is_owner,
            is_admin: val.is_admin,
            auth_token: val.auth_token,
            created_at: val.created_at,
            active: val.active,
            is_password_set: val.password_encrypted.is_some(),

            // when a user created by registration, password_encrypted is set
            // when a user created by SSO, password_encrypted is not set
            // so, we can determine if a user is SSO user by checking if password_encrypted is set
            is_sso_user: val.password_encrypted.is_none(),
        }
    }
}

#[async_trait::async_trait]
trait UserGroupExt {
    async fn new(db: DbConn, val: UserGroupDAO) -> Result<UserGroup>;
}

#[async_trait::async_trait]
impl UserGroupExt for UserGroup {
    async fn new(db: DbConn, val: UserGroupDAO) -> Result<UserGroup> {
        let mut members = Vec::new();
        for x in db.list_user_group_memberships(val.id, None).await? {
            members.push(UserGroupMembership {
                is_group_admin: x.is_group_admin,
                created_at: x.created_at,
                updated_at: x.updated_at,
                user: UserValue::UserSecured(UserSecured::new(
                    db.clone(),
                    db.get_user(x.user_id)
                        .await?
                        .context("User doesn't exists")?,
                )),
            });
        }

        members.sort_by_key(|x| (!x.is_group_admin, x.updated_at));

        Ok(UserGroup {
            id: val.id.as_id(),
            name: val.name,
            created_at: val.created_at,
            updated_at: val.updated_at,
            members,
        })
    }
}
