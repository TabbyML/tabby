pub mod analytic;
pub mod auth;
pub mod constants;
pub mod email;
pub mod job;
pub mod license;
pub mod repository;
pub mod setting;
pub mod user_event;
pub mod worker;

use std::sync::Arc;

use auth::{
    AuthenticationService, Invitation, RefreshTokenResponse, RegisterResponse, TokenAuthResponse,
    User,
};
use base64::Engine;
use chrono::{DateTime, Utc};
use job::{JobRun, JobService};
use juniper::{
    graphql_object, graphql_value, EmptySubscription, FieldError, GraphQLObject, IntoFieldError,
    Object, RootNode, ScalarValue, Value, ID,
};
use tabby_common::api::{code::CodeSearch, event::EventLogger};
use tracing::error;
use validator::{Validate, ValidationErrors};
use worker::{Worker, WorkerService};

use self::{
    analytic::{AnalyticService, CompletionStats, DiskUsageStats},
    auth::{
        JWTPayload, OAuthCredential, OAuthProvider, PasswordChangeInput, PasswordResetInput,
        RequestInvitationInput, RequestPasswordResetEmailInput, UpdateOAuthCredentialInput,
    },
    email::{EmailService, EmailSetting, EmailSettingInput},
    job::JobStats,
    license::{IsLicenseValid, LicenseInfo, LicenseService, LicenseType},
    repository::{FileEntrySearchResult, Repository, RepositoryKind, RepositoryService},
    setting::{
        NetworkSetting, NetworkSettingInput, SecuritySetting, SecuritySettingInput, SettingService,
    },
    user_event::{UserEvent, UserEventService},
};
use crate::{
    env,
    juniper::relay::{self, Connection},
};

pub trait ServiceLocator: Send + Sync {
    fn auth(&self) -> Arc<dyn AuthenticationService>;
    fn worker(&self) -> Arc<dyn WorkerService>;
    fn code(&self) -> Arc<dyn CodeSearch>;
    fn logger(&self) -> Arc<dyn EventLogger>;
    fn job(&self) -> Arc<dyn JobService>;
    fn repository(&self) -> Arc<dyn RepositoryService>;
    fn email(&self) -> Arc<dyn EmailService>;
    fn setting(&self) -> Arc<dyn SettingService>;
    fn license(&self) -> Arc<dyn LicenseService>;
    fn analytic(&self) -> Arc<dyn AnalyticService>;
    fn user_event(&self) -> Arc<dyn UserEventService>;
}

pub struct Context {
    pub claims: Option<auth::JWTPayload>,
    pub locator: Arc<dyn ServiceLocator>,
}

// To make our context usable by Juniper, we have to implement a marker trait.
impl juniper::Context for Context {}

pub type Result<T, E = CoreError> = std::result::Result<T, E>;

#[derive(thiserror::Error, Debug)]
pub enum CoreError {
    #[error("{0}")]
    Unauthorized(&'static str),

    #[error("{0}")]
    Forbidden(&'static str),

    #[error("Invalid ID")]
    InvalidID,

    #[error("Invalid input parameters")]
    InvalidInput(#[from] ValidationErrors),

    #[error("Email is not configured")]
    EmailNotConfigured,

    #[error("{0}")]
    InvalidLicense(&'static str),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl<S: ScalarValue> IntoFieldError<S> for CoreError {
    fn into_field_error(self) -> FieldError<S> {
        match self {
            Self::Forbidden(msg) => FieldError::new(msg, graphql_value!({"code": "FORBIDDEN"})),
            Self::Unauthorized(msg) => {
                FieldError::new(msg, graphql_value!({"code": "UNAUTHORIZED"}))
            }
            Self::InvalidInput(errors) => from_validation_errors(errors),
            _ => self.into(),
        }
    }
}

fn check_claims(ctx: &Context) -> Result<&JWTPayload, CoreError> {
    ctx.claims
        .as_ref()
        .ok_or(CoreError::Unauthorized("You're not logged in"))
}

async fn check_admin(ctx: &Context) -> Result<(), CoreError> {
    let user = check_user(ctx).await?;
    if !user.is_admin {
        return Err(CoreError::Forbidden("You must be admin to proceed"));
    }

    Ok(())
}

async fn check_user(ctx: &Context) -> Result<User, CoreError> {
    let claims = check_claims(ctx)?;
    let user = ctx.locator.auth().get_user(&claims.sub).await?;
    Ok(user)
}

async fn check_license(ctx: &Context, license_type: &[LicenseType]) -> Result<(), CoreError> {
    let license = ctx.locator.license().read().await?;

    if !license_type.contains(&license.r#type) {
        return Err(CoreError::InvalidLicense(
            "Your plan doesn't include support for this feature.",
        ));
    }

    license.ensure_valid_license()
}

#[derive(Default)]
pub struct Query;

#[graphql_object(context = Context)]
impl Query {
    async fn workers(ctx: &Context) -> Result<Vec<Worker>> {
        check_admin(ctx).await?;
        let workers = ctx.locator.worker().list().await;
        Ok(workers)
    }

    async fn registration_token(ctx: &Context) -> Result<String> {
        check_admin(ctx).await?;
        ctx.locator.worker().read_registration_token().await
    }

    async fn me(ctx: &Context) -> Result<User> {
        let claims = check_claims(ctx)?;
        ctx.locator.auth().get_user(&claims.sub).await
    }

    async fn users(
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<User>> {
        check_admin(ctx).await?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .auth()
                    .list_users(after, before, first, last)
                    .await
            },
        )
        .await
    }

    async fn invitations(
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<Invitation>> {
        check_admin(ctx).await?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .auth()
                    .list_invitations(after, before, first, last)
                    .await
            },
        )
        .await
    }

    async fn github_repository_providers(
        ctx: &Context,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<repository::GithubRepositoryProvider>> {
        check_admin(ctx).await?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .repository()
                    .github()
                    .list_providers(ids.unwrap_or_default(), after, before, first, last)
                    .await
            },
        )
        .await
    }

    async fn github_repositories(
        ctx: &Context,
        provider_ids: Vec<ID>,
        active: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<repository::GithubProvidedRepository>> {
        check_admin(ctx).await?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .repository()
                    .github()
                    .list_repositories(provider_ids, active, after, before, first, last)
                    .await
            },
        )
        .await
    }

    async fn gitlab_repository_providers(
        ctx: &Context,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<repository::GitlabRepositoryProvider>> {
        check_admin(ctx).await?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .repository()
                    .gitlab()
                    .list_providers(ids.unwrap_or_default(), after, before, first, last)
                    .await
            },
        )
        .await
    }

    async fn gitlab_repositories(
        ctx: &Context,
        provider_ids: Vec<ID>,
        active: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<repository::GitlabProvidedRepository>> {
        check_admin(ctx).await?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .repository()
                    .gitlab()
                    .list_repositories(provider_ids, active, after, before, first, last)
                    .await
            },
        )
        .await
    }

    async fn job_runs(
        ctx: &Context,
        ids: Option<Vec<ID>>,
        jobs: Option<Vec<String>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<JobRun>> {
        check_admin(ctx).await?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .job()
                    .list(ids, jobs, after, before, first, last)
                    .await
            },
        )
        .await
    }

    async fn job_run_stats(ctx: &Context, jobs: Option<Vec<String>>) -> Result<JobStats> {
        ctx.locator.job().compute_stats(jobs).await
    }

    async fn email_setting(ctx: &Context) -> Result<Option<EmailSetting>> {
        check_admin(ctx).await?;
        ctx.locator.email().read_setting().await
    }

    async fn network_setting(ctx: &Context) -> Result<NetworkSetting> {
        check_admin(ctx).await?;
        ctx.locator.setting().read_network_setting().await
    }

    async fn security_setting(ctx: &Context) -> Result<SecuritySetting> {
        check_admin(ctx).await?;
        ctx.locator.setting().read_security_setting().await
    }

    async fn git_repositories(
        &self,
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<repository::GitRepository>> {
        check_admin(ctx).await?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .repository()
                    .git()
                    .list(after, before, first, last)
                    .await
            },
        )
        .await
    }

    async fn repository_search(
        ctx: &Context,
        kind: RepositoryKind,
        id: ID,
        pattern: String,
    ) -> Result<Vec<FileEntrySearchResult>> {
        check_claims(ctx)?;
        ctx.locator
            .repository()
            .search_files(&kind, &id, &pattern, 40)
            .await
    }

    async fn oauth_credential(
        ctx: &Context,
        provider: OAuthProvider,
    ) -> Result<Option<OAuthCredential>> {
        check_admin(ctx).await?;
        ctx.locator.auth().read_oauth_credential(provider).await
    }

    async fn oauth_callback_url(ctx: &Context, provider: OAuthProvider) -> Result<String> {
        check_admin(ctx).await?;
        ctx.locator.auth().oauth_callback_url(provider).await
    }

    async fn server_info(ctx: &Context) -> Result<ServerInfo> {
        Ok(ServerInfo {
            is_admin_initialized: ctx.locator.auth().is_admin_initialized().await?,
            is_chat_enabled: ctx.locator.worker().is_chat_enabled().await?,
            is_email_configured: ctx.locator.email().read_setting().await?.is_some(),
            allow_self_signup: ctx.locator.auth().allow_self_signup().await?,
            is_demo_mode: env::demo_mode(),
        })
    }

    async fn license(ctx: &Context) -> Result<LicenseInfo> {
        ctx.locator.license().read().await
    }

    // FIXME(meng): This is a temporary solution to expose the list of jobs, we should consider switching to a enum based approach.
    async fn jobs() -> Result<Vec<String>> {
        Ok(vec![
            "scheduler",
            "import_github_repositories",
            "import_gitlab_repositories",
        ]
        .into_iter()
        .map(Into::into)
        .collect())
    }

    async fn daily_stats_in_past_year(
        ctx: &Context,
        users: Option<Vec<ID>>,
    ) -> Result<Vec<CompletionStats>> {
        let users = users.unwrap_or_default();
        check_analytic_access(ctx, &users).await?;
        ctx.locator.analytic().daily_stats_in_past_year(users).await
    }

    async fn daily_stats(
        ctx: &Context,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        users: Option<Vec<ID>>,
        languages: Option<Vec<analytic::Language>>,
    ) -> Result<Vec<CompletionStats>> {
        let users = users.unwrap_or_default();
        check_analytic_access(ctx, &users).await?;
        ctx.locator
            .analytic()
            .daily_stats(start, end, users, languages.unwrap_or_default())
            .await
    }

    async fn user_events(
        ctx: &Context,

        // pagination arguments
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,

        // filter arguments
        users: Option<Vec<ID>>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Connection<UserEvent>> {
        check_admin(ctx).await?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .user_event()
                    .list(
                        after,
                        before,
                        first,
                        last,
                        users.unwrap_or_default(),
                        start,
                        end,
                    )
                    .await
            },
        )
        .await
    }

    async fn disk_usage_stats(ctx: &Context) -> Result<DiskUsageStats> {
        check_admin(ctx).await?;
        ctx.locator.analytic().disk_usage_stats().await
    }

    async fn repository_list(ctx: &Context) -> Result<Vec<Repository>> {
        check_admin(ctx).await?;
        ctx.locator.repository().repository_list().await
    }
}

#[derive(GraphQLObject)]
pub struct ServerInfo {
    is_admin_initialized: bool,
    is_chat_enabled: bool,
    is_email_configured: bool,
    allow_self_signup: bool,
    is_demo_mode: bool,
}

#[derive(Default)]
pub struct Mutation;

#[graphql_object(context = Context)]
impl Mutation {
    async fn reset_registration_token(ctx: &Context) -> Result<String> {
        check_admin(ctx).await?;
        ctx.locator.worker().reset_registration_token().await
    }

    async fn request_invitation_email(
        ctx: &Context,
        input: RequestInvitationInput,
    ) -> Result<Invitation> {
        input.validate()?;
        ctx.locator.auth().request_invitation_email(input).await
    }

    async fn request_password_reset_email(
        ctx: &Context,
        input: RequestPasswordResetEmailInput,
    ) -> Result<bool> {
        input.validate()?;
        ctx.locator
            .auth()
            .request_password_reset_email(input.email)
            .await?;
        Ok(true)
    }

    async fn password_reset(ctx: &Context, input: PasswordResetInput) -> Result<bool> {
        input.validate()?;
        ctx.locator
            .auth()
            .password_reset(&input.code, &input.password1)
            .await?;
        Ok(true)
    }

    async fn password_change(ctx: &Context, input: PasswordChangeInput) -> Result<bool> {
        let claims = check_claims(ctx)?;
        input.validate()?;
        ctx.locator
            .auth()
            .update_user_password(
                &claims.sub,
                input.old_password.as_deref(),
                &input.new_password1,
            )
            .await?;
        Ok(true)
    }

    async fn reset_user_auth_token(ctx: &Context) -> Result<bool> {
        let claims = check_claims(ctx)?;
        ctx.locator
            .auth()
            .reset_user_auth_token(&claims.sub)
            .await?;
        Ok(true)
    }

    async fn logout_all_sessions(ctx: &Context) -> Result<bool> {
        let claims = check_claims(ctx)?;
        ctx.locator.auth().logout_all_sessions(&claims.sub).await?;
        Ok(true)
    }

    async fn update_user_active(ctx: &Context, id: ID, active: bool) -> Result<bool> {
        check_admin(ctx).await?;
        if ctx.claims.as_ref().is_some_and(|c| c.sub == id) {
            return Err(CoreError::Forbidden(
                "You cannot change your own active status",
            ));
        }
        ctx.locator.auth().update_user_active(&id, active).await?;
        Ok(true)
    }

    async fn update_user_role(ctx: &Context, id: ID, is_admin: bool) -> Result<bool> {
        check_admin(ctx).await?;
        if ctx.claims.as_ref().is_some_and(|c| c.sub == id) {
            return Err(CoreError::Forbidden("You cannot update your own role"));
        }
        ctx.locator.auth().update_user_role(&id, is_admin).await?;
        Ok(true)
    }

    async fn upload_user_avatar_base64(
        ctx: &Context,
        id: ID,
        avatar_base64: Option<String>,
    ) -> Result<bool> {
        let claims = check_claims(ctx)?;
        if claims.sub != id {
            return Err(CoreError::Unauthorized(
                "You cannot change another user's avatar",
            ));
        }
        // ast-grep-ignore: use-schema-result
        use anyhow::Context;
        let avatar = avatar_base64
            .map(|avatar| base64::prelude::BASE64_STANDARD.decode(avatar.as_bytes()))
            .transpose()
            .context("avatar is not valid base64 string")?
            .map(Vec::into_boxed_slice);
        ctx.locator.auth().update_user_avatar(&id, avatar).await?;
        Ok(true)
    }

    async fn register(
        ctx: &Context,
        email: String,
        password1: String,
        password2: String,
        invitation_code: Option<String>,
    ) -> Result<RegisterResponse> {
        let input = auth::RegisterInput {
            email,
            password1,
            password2,
        };
        input.validate()?;

        ctx.locator
            .auth()
            .register(input.email, input.password1, invitation_code)
            .await
    }

    async fn token_auth(
        ctx: &Context,
        email: String,
        password: String,
    ) -> Result<TokenAuthResponse> {
        let input = auth::TokenAuthInput { email, password };
        input.validate()?;
        ctx.locator
            .auth()
            .token_auth(input.email, input.password)
            .await
    }

    async fn verify_token(ctx: &Context, token: String) -> Result<bool> {
        ctx.locator.auth().verify_access_token(&token).await?;
        Ok(true)
    }

    async fn refresh_token(ctx: &Context, refresh_token: String) -> Result<RefreshTokenResponse> {
        ctx.locator.auth().refresh_token(refresh_token).await
    }

    async fn create_invitation(ctx: &Context, email: String) -> Result<ID> {
        check_admin(ctx).await?;
        let invitation = ctx.locator.auth().create_invitation(email.clone()).await?;
        Ok(invitation.id)
    }

    async fn send_test_email(ctx: &Context, to: String) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator.email().send_test(to).await?;
        Ok(true)
    }

    async fn create_git_repository(ctx: &Context, name: String, git_url: String) -> Result<ID> {
        check_admin(ctx).await?;
        let input = repository::CreateGitRepositoryInput { name, git_url };
        input.validate()?;
        ctx.locator
            .repository()
            .git()
            .create(input.name, input.git_url)
            .await
    }

    async fn delete_git_repository(ctx: &Context, id: ID) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator.repository().git().delete(&id).await
    }

    async fn update_git_repository(
        ctx: &Context,
        id: ID,
        name: String,
        git_url: String,
    ) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator
            .repository()
            .git()
            .update(&id, name, git_url)
            .await
    }

    async fn delete_invitation(ctx: &Context, id: ID) -> Result<ID> {
        check_admin(ctx).await?;
        ctx.locator.auth().delete_invitation(&id).await
    }

    async fn update_oauth_credential(
        ctx: &Context,
        input: UpdateOAuthCredentialInput,
    ) -> Result<bool> {
        check_admin(ctx).await?;
        check_license(ctx, &[LicenseType::Enterprise]).await?;
        input.validate()?;
        ctx.locator.auth().update_oauth_credential(input).await?;
        Ok(true)
    }

    async fn delete_oauth_credential(ctx: &Context, provider: OAuthProvider) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator.auth().delete_oauth_credential(provider).await?;
        Ok(true)
    }

    async fn update_email_setting(ctx: &Context, input: EmailSettingInput) -> Result<bool> {
        check_admin(ctx).await?;
        input.validate()?;
        ctx.locator.email().update_setting(input).await?;
        Ok(true)
    }

    async fn update_security_setting(ctx: &Context, input: SecuritySettingInput) -> Result<bool> {
        check_admin(ctx).await?;
        check_license(ctx, &[LicenseType::Enterprise]).await?;
        input.validate()?;
        ctx.locator.setting().update_security_setting(input).await?;
        Ok(true)
    }

    async fn update_network_setting(ctx: &Context, input: NetworkSettingInput) -> Result<bool> {
        check_admin(ctx).await?;
        input.validate()?;
        ctx.locator.setting().update_network_setting(input).await?;
        Ok(true)
    }

    async fn delete_email_setting(ctx: &Context) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator.email().delete_setting().await?;
        Ok(true)
    }

    async fn upload_license(ctx: &Context, license: String) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator.license().update(license).await?;
        Ok(true)
    }

    async fn reset_license(ctx: &Context) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator.license().reset().await?;
        Ok(true)
    }

    async fn create_github_repository_provider(
        ctx: &Context,
        input: repository::CreateRepositoryProviderInput,
    ) -> Result<ID> {
        check_admin(ctx).await?;
        input.validate()?;
        let id = ctx
            .locator
            .repository()
            .github()
            .create_provider(input.display_name, input.access_token)
            .await?;
        Ok(id)
    }

    async fn delete_github_repository_provider(ctx: &Context, id: ID) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator
            .repository()
            .github()
            .delete_provider(id)
            .await?;
        Ok(true)
    }

    async fn update_github_repository_provider(
        ctx: &Context,
        input: repository::UpdateRepositoryProviderInput,
    ) -> Result<bool> {
        check_admin(ctx).await?;
        input.validate()?;
        ctx.locator
            .repository()
            .github()
            .update_provider(input.id, input.display_name, input.access_token)
            .await?;
        Ok(true)
    }

    async fn update_github_provided_repository_active(
        ctx: &Context,
        id: ID,
        active: bool,
    ) -> Result<bool> {
        ctx.locator
            .repository()
            .github()
            .update_repository_active(id, active)
            .await?;
        Ok(true)
    }

    async fn create_gitlab_repository_provider(
        ctx: &Context,
        input: repository::CreateRepositoryProviderInput,
    ) -> Result<ID> {
        check_admin(ctx).await?;
        input.validate()?;
        let id = ctx
            .locator
            .repository()
            .gitlab()
            .create_provider(input.display_name, input.access_token)
            .await?;
        Ok(id)
    }

    async fn delete_gitlab_repository_provider(ctx: &Context, id: ID) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator
            .repository()
            .gitlab()
            .delete_provider(id)
            .await?;
        Ok(true)
    }

    async fn update_gitlab_repository_provider(
        ctx: &Context,
        input: repository::UpdateRepositoryProviderInput,
    ) -> Result<bool> {
        check_admin(ctx).await?;
        input.validate()?;
        ctx.locator
            .repository()
            .gitlab()
            .update_provider(input.id, input.display_name, input.access_token)
            .await?;
        Ok(true)
    }

    async fn update_gitlab_provided_repository_active(
        ctx: &Context,
        id: ID,
        active: bool,
    ) -> Result<bool> {
        ctx.locator
            .repository()
            .gitlab()
            .update_repository_active(id, active)
            .await?;
        Ok(true)
    }
}

async fn check_analytic_access(ctx: &Context, users: &[ID]) -> Result<(), CoreError> {
    let user = check_user(ctx).await?;
    if users.is_empty() && !user.is_admin {
        return Err(CoreError::Forbidden(
            "You must be admin to read other users' data",
        ));
    }

    if !user.is_admin {
        for id in users {
            if user.id != *id {
                return Err(CoreError::Forbidden(
                    "You must be admin to read other users' data",
                ));
            }
        }
    }

    Ok(())
}

fn from_validation_errors<S: ScalarValue>(error: ValidationErrors) -> FieldError<S> {
    let errors = error
        .field_errors()
        .into_iter()
        .flat_map(|(_, errs)| errs)
        .cloned()
        .map(|err| {
            let mut obj = Object::with_capacity(2);
            obj.add_field("path", Value::scalar(err.code.to_string()));
            obj.add_field(
                "message",
                Value::scalar(err.message.unwrap_or_default().to_string()),
            );
            obj.into()
        })
        .collect::<Vec<_>>();
    let mut error = Object::with_capacity(1);
    error.add_field("errors", Value::list(errors));

    let mut ext = Object::with_capacity(1);
    ext.add_field("validation-errors", error.into());

    FieldError::new("Invalid input parameters", ext.into())
}

pub type Schema = RootNode<'static, Query, Mutation, EmptySubscription<Context>>;

pub fn create_schema() -> Schema {
    Schema::new(Query, Mutation, EmptySubscription::new())
}
