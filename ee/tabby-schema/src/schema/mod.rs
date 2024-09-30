pub mod access_policy;
pub mod analytic;
pub mod auth;
pub mod constants;
pub mod context;
pub mod email;
pub mod integration;
pub mod interface;
pub mod job;
pub mod license;
pub mod repository;
pub mod setting;
pub mod thread;
pub mod user_event;
pub mod user_group;
pub mod web_documents;
pub mod worker;

use std::sync::Arc;

use access_policy::{AccessPolicyService, SourceIdAccessPolicy};
use auth::{
    AuthenticationService, Invitation, RefreshTokenResponse, RegisterResponse, TokenAuthResponse,
    UserSecured,
};
use base64::Engine;
use chrono::{DateTime, Utc};
use context::{ContextInfo, ContextService};
use interface::UserValue;
use job::{JobRun, JobService};
use juniper::{
    graphql_object, graphql_subscription, graphql_value, FieldError, GraphQLObject, IntoFieldError,
    Object, RootNode, ScalarValue, Value, ID,
};
use repository::RepositoryGrepOutput;
use tabby_common::api::{code::CodeSearch, event::EventLogger};
use thread::{CreateThreadAndRunInput, CreateThreadRunInput, ThreadRunStream, ThreadService};
use tracing::{error, warn};
use user_group::{
    CreateUserGroupInput, UpsertUserGroupMembershipInput, UserGroup, UserGroupService,
};
use validator::{Validate, ValidationErrors};
use worker::WorkerService;

use self::{
    analytic::{AnalyticService, CompletionStats, DiskUsageStats},
    auth::{
        JWTPayload, OAuthCredential, OAuthProvider, PasswordChangeInput, PasswordResetInput,
        RequestInvitationInput, RequestPasswordResetEmailInput, UpdateOAuthCredentialInput,
    },
    email::{EmailService, EmailSetting, EmailSettingInput},
    integration::{Integration, IntegrationKind, IntegrationService},
    job::JobStats,
    license::{IsLicenseValid, LicenseInfo, LicenseService, LicenseType},
    repository::{
        CreateIntegrationInput, FileEntrySearchResult, ProvidedRepository, Repository,
        RepositoryKind, RepositoryService, UpdateIntegrationInput,
    },
    setting::{
        NetworkSetting, NetworkSettingInput, SecuritySetting, SecuritySettingInput, SettingService,
    },
    user_event::{UserEvent, UserEventService},
    web_documents::{CreateCustomDocumentInput, CustomWebDocument, WebDocumentService},
};
use crate::{
    env,
    juniper::relay::{self, query_async, Connection},
    web_documents::{PresetWebDocument, SetPresetDocumentActiveInput},
};

pub trait ServiceLocator: Send + Sync {
    fn auth(&self) -> Arc<dyn AuthenticationService>;
    fn worker(&self) -> Arc<dyn WorkerService>;
    fn code(&self) -> Arc<dyn CodeSearch>;
    fn logger(&self) -> Arc<dyn EventLogger>;
    fn job(&self) -> Arc<dyn JobService>;
    fn repository(&self) -> Arc<dyn RepositoryService>;
    fn integration(&self) -> Arc<dyn IntegrationService>;
    fn email(&self) -> Arc<dyn EmailService>;
    fn setting(&self) -> Arc<dyn SettingService>;
    fn license(&self) -> Arc<dyn LicenseService>;
    fn analytic(&self) -> Arc<dyn AnalyticService>;
    fn user_event(&self) -> Arc<dyn UserEventService>;
    fn web_documents(&self) -> Arc<dyn WebDocumentService>;
    fn thread(&self) -> Arc<dyn ThreadService>;
    fn context(&self) -> Arc<dyn ContextService>;
    fn user_group(&self) -> Arc<dyn UserGroupService>;
    fn access_policy(&self) -> Arc<dyn AccessPolicyService>;
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

    #[error("{0}")]
    NotFound(&'static str),

    #[error("Invalid ID")]
    InvalidID,

    #[error("Invalid input parameters")]
    InvalidInput(#[from] ValidationErrors),

    #[error("SMTP is not configured")]
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
            Self::NotFound(msg) => FieldError::new(msg, graphql_value!({"code": "NOT_FOUND"})),
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

async fn check_user(ctx: &Context) -> Result<UserSecured, CoreError> {
    check_user_and_auth_token(ctx, false).await
}

async fn check_user_allow_auth_token(ctx: &Context) -> Result<UserSecured, CoreError> {
    check_user_and_auth_token(ctx, true).await
}

async fn check_user_and_auth_token(
    ctx: &Context,
    allow_auth_token: bool,
) -> Result<UserSecured, CoreError> {
    let claims = check_claims(ctx)?;
    if !allow_auth_token && claims.is_generated_from_auth_token {
        return Err(CoreError::Forbidden(
            "Invoking this API with an auth token is not allowed",
        ));
    }
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
    async fn registration_token(ctx: &Context) -> Result<String> {
        check_admin(ctx).await?;
        ctx.locator.worker().read_registration_token().await
    }

    async fn me(ctx: &Context) -> Result<UserSecured> {
        check_user_allow_auth_token(ctx).await
    }

    /// List users, accessible for all login users.
    async fn users(
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<UserValue>> {
        check_user(ctx).await?;
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
                    .map(|users| users.into_iter().map(UserValue::UserSecured).collect())
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

    /// Search files that matches the pattern in the repository.
    async fn repository_search(
        ctx: &Context,
        kind: RepositoryKind,
        id: ID,
        rev: Option<String>,
        pattern: String,
    ) -> Result<Vec<FileEntrySearchResult>> {
        let user = check_user(ctx).await?;
        ctx.locator
            .repository()
            .search_files(&user.policy, &kind, &id, rev.as_deref(), &pattern, 40)
            .await
    }

    /// File content search with a grep-like experience.
    ///
    /// Syntax:
    ///
    /// 1. Unprefixed text will be treated as a regex pattern for file content search.
    /// 2. 'f:' to search by file name with a regex pattern.
    /// 3. 'lang:' to search by file language.
    /// 4. All tokens can be negated by prefixing them with '-'.
    ///
    /// Examples:
    /// * `f:schema -lang:rust fn`
    /// * `func_name lang:go`
    async fn repository_grep(
        ctx: &Context,
        kind: RepositoryKind,
        id: ID,
        rev: Option<String>,
        query: String,
    ) -> Result<RepositoryGrepOutput> {
        let user = check_user(ctx).await?;

        let start_time = chrono::offset::Utc::now();
        let files = ctx
            .locator
            .repository()
            .grep(&user.policy, &kind, &id, rev.as_deref(), &query, 40)
            .await?;
        let end_time = chrono::offset::Utc::now();
        let elapsed_ms = (end_time - start_time).num_milliseconds() as i32;
        Ok(RepositoryGrepOutput { files, elapsed_ms })
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
            is_demo_mode: env::is_demo_mode(),
        })
    }

    async fn license(ctx: &Context) -> Result<LicenseInfo> {
        ctx.locator.license().read().await
    }

    // FIXME(meng): This is a temporary solution to expose the list of jobs, we should consider switching to a enum based approach.
    async fn jobs() -> Result<Vec<String>> {
        Ok(
            vec!["scheduler_git", "scheduler_github_gitlab", "web_crawler"]
                .into_iter()
                .map(Into::into)
                .collect(),
        )
    }

    async fn daily_stats_in_past_year(
        ctx: &Context,
        users: Option<Vec<ID>>,
    ) -> Result<Vec<CompletionStats>> {
        let users = users.unwrap_or_default();
        let user = check_user(ctx).await?;
        user.policy.check_read_analytic(&users)?;
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
        let user = check_user(ctx).await?;
        user.policy.check_read_analytic(&users)?;
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
        let user = check_user(ctx).await?;

        ctx.locator
            .repository()
            .repository_list(Some(&user.policy))
            .await
    }

    async fn context_info(ctx: &Context) -> Result<ContextInfo> {
        let user = check_user(ctx).await?;
        ctx.locator.context().read(Some(&user.policy)).await
    }

    async fn integrations(
        ctx: &Context,
        ids: Option<Vec<ID>>,
        kind: Option<IntegrationKind>,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<Integration>> {
        check_admin(ctx).await?;
        query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .integration()
                    .list_integrations(ids, kind, after, before, first, last)
                    .await
            },
        )
        .await
    }

    async fn integrated_repositories(
        ctx: &Context,
        ids: Option<Vec<ID>>,
        kind: Option<IntegrationKind>,
        active: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<ProvidedRepository>> {
        check_admin(ctx).await?;
        query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .repository()
                    .third_party()
                    .list_repositories_with_filter(ids, kind, active, after, before, first, last)
                    .await
            },
        )
        .await
    }

    async fn threads(
        ctx: &Context,
        ids: Option<Vec<ID>>,
        is_ephemeral: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<thread::Thread>> {
        check_user(ctx).await?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .thread()
                    .list(ids.as_deref(), is_ephemeral, after, before, first, last)
                    .await
            },
        )
        .await
    }

    /// Read thread messages by thread ID.
    ///
    /// Thread is public within an instance, so no need to check for ownership.
    async fn thread_messages(
        ctx: &Context,
        thread_id: ID,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<thread::Message>> {
        check_user(ctx).await?;

        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .thread()
                    .list_thread_messages(&thread_id, after, before, first, last)
                    .await
            },
        )
        .await
    }

    async fn custom_web_documents(
        ctx: &Context,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> Result<Connection<CustomWebDocument>> {
        check_admin(ctx).await?;
        query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .web_documents()
                    .list_custom_web_documents(ids, after, before, first, last)
                    .await
            },
        )
        .await
    }
    async fn preset_web_documents(
        ctx: &Context,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
        is_active: Option<bool>,
    ) -> Result<Connection<PresetWebDocument>> {
        check_admin(ctx).await?;
        query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                ctx.locator
                    .web_documents()
                    .list_preset_web_documents(ids, after, before, first, last, is_active)
                    .await
            },
        )
        .await
    }

    /// List user groups.
    async fn user_groups(ctx: &Context) -> Result<Vec<UserGroup>> {
        check_user(ctx).await?;
        ctx.locator.user_group().list().await
    }

    async fn source_id_access_policies(
        ctx: &Context,
        source_id: String,
    ) -> Result<SourceIdAccessPolicy> {
        check_admin(ctx).await?;
        let read = ctx
            .locator
            .access_policy()
            .list_source_id_read_access(&source_id)
            .await?;

        Ok(SourceIdAccessPolicy { source_id, read })
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

    async fn generate_reset_password_url(ctx: &Context, user_id: ID) -> Result<String> {
        check_admin(ctx).await?;
        ctx.locator
            .auth()
            .generate_reset_password_url(&user_id)
            .await
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

    async fn update_user_name(ctx: &Context, id: ID, name: String) -> Result<bool> {
        let claims = check_claims(ctx)?;
        if claims.sub != id {
            return Err(CoreError::Unauthorized(
                "You cannot change another user's name",
            ));
        }
        let input = auth::UpdateUserNameInput { name };
        input.validate()?;
        ctx.locator.auth().update_user_name(&id, input.name).await?;
        Ok(true)
    }

    async fn register(
        ctx: &Context,
        email: String,
        password1: String,
        password2: String,
        invitation_code: Option<String>,
        name: String,
    ) -> Result<RegisterResponse> {
        let input = auth::RegisterInput {
            email,
            password1,
            password2,
        };
        input.validate()?;

        ctx.locator
            .auth()
            .register(input.email, input.password1, invitation_code, Some(name))
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

    async fn create_integration(ctx: &Context, input: CreateIntegrationInput) -> Result<ID> {
        check_admin(ctx).await?;
        input.validate()?;
        let id = ctx
            .locator
            .integration()
            .create_integration(
                input.kind,
                input.display_name,
                input.access_token,
                input.api_base,
            )
            .await?;
        Ok(id)
    }

    async fn update_integration(ctx: &Context, input: UpdateIntegrationInput) -> Result<bool> {
        check_admin(ctx).await?;
        input.validate()?;
        ctx.locator
            .integration()
            .update_integration(
                input.id,
                input.kind,
                input.display_name,
                input.access_token,
                input.api_base,
            )
            .await?;
        Ok(true)
    }

    async fn delete_integration(ctx: &Context, id: ID, kind: IntegrationKind) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator
            .integration()
            .delete_integration(id, kind)
            .await?;
        Ok(true)
    }

    async fn update_integrated_repository_active(
        ctx: &Context,
        id: ID,
        active: bool,
    ) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator
            .repository()
            .third_party()
            .update_repository_active(id, active)
            .await?;
        Ok(true)
    }

    /// Trigger a job run given its param string.
    async fn trigger_job_run(ctx: &Context, command: String) -> Result<ID> {
        check_admin(ctx).await?;
        ctx.locator.job().trigger(command).await
    }

    /// Delete pair of user message and bot response in a thread.
    async fn delete_thread_message_pair(
        ctx: &Context,
        thread_id: ID,
        user_message_id: ID,
        assistant_message_id: ID,
    ) -> Result<bool> {
        let user = check_user_allow_auth_token(ctx).await?;
        let svc = ctx.locator.thread();
        let Some(thread) = svc.get(&thread_id).await? else {
            return Err(CoreError::NotFound("Thread not found"));
        };

        user.policy.check_delete_thread_messages(&thread.user_id)?;

        ctx.locator
            .thread()
            .delete_thread_message_pair(&thread_id, &user_message_id, &assistant_message_id)
            .await?;
        Ok(true)
    }

    /// Turn on persisted status for a thread.
    async fn set_thread_persisted(ctx: &Context, thread_id: ID) -> Result<bool> {
        let user = check_user(ctx).await?;
        let svc = ctx.locator.thread();
        let Some(thread) = svc.get(&thread_id).await? else {
            return Err(CoreError::NotFound("Thread not found"));
        };

        user.policy
            .check_update_thread_persistence(&thread.user_id)?;

        ctx.locator.thread().set_persisted(&thread_id).await?;
        Ok(true)
    }

    async fn create_custom_document(ctx: &Context, input: CreateCustomDocumentInput) -> Result<ID> {
        check_admin(ctx).await?;
        input.validate()?;
        let id = ctx
            .locator
            .web_documents()
            .create_custom_web_document(input.name, input.url)
            .await?;
        Ok(id)
    }

    async fn delete_custom_document(ctx: &Context, id: ID) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator
            .web_documents()
            .delete_custom_web_document(id)
            .await?;
        Ok(true)
    }

    async fn set_preset_document_active(
        ctx: &Context,
        input: SetPresetDocumentActiveInput,
    ) -> Result<bool> {
        check_admin(ctx).await?;
        input.validate()?;
        ctx.locator
            .web_documents()
            .set_preset_web_documents_active(input.id, input.active)
            .await?;
        Ok(true)
    }

    async fn create_user_group(ctx: &Context, input: CreateUserGroupInput) -> Result<ID> {
        check_admin(ctx).await?;
        input.validate()?;
        let id = ctx.locator.user_group().create(&input).await?;
        Ok(id)
    }

    async fn delete_user_group(ctx: &Context, id: ID) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator.user_group().delete(&id).await?;
        Ok(true)
    }

    async fn upsert_user_group_membership(
        ctx: &Context,
        input: UpsertUserGroupMembershipInput,
    ) -> Result<bool> {
        let user = check_user(ctx).await?;
        user.policy
            .check_upsert_user_group_membership(&input)
            .await?;

        input.validate()?;
        ctx.locator.user_group().upsert_membership(&input).await?;
        Ok(true)
    }

    async fn delete_user_group_membership(
        ctx: &Context,
        user_group_id: ID,
        user_id: ID,
    ) -> Result<bool> {
        let user = check_user(ctx).await?;
        user.policy
            .check_delete_user_group_membership(&user_group_id, &user_id)
            .await?;

        ctx.locator
            .user_group()
            .delete_membership(&user_group_id, &user_id)
            .await?;
        Ok(true)
    }

    async fn grant_source_id_read_access(
        ctx: &Context,
        source_id: String,
        user_group_id: ID,
    ) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator
            .access_policy()
            .grant_source_id_read_access(&source_id, &user_group_id)
            .await?;
        Ok(true)
    }

    async fn revoke_source_id_read_access(
        ctx: &Context,
        source_id: String,
        user_group_id: ID,
    ) -> Result<bool> {
        check_admin(ctx).await?;
        ctx.locator
            .access_policy()
            .revoke_source_id_read_access(&source_id, &user_group_id)
            .await?;
        Ok(true)
    }
}

fn from_validation_errors<S: ScalarValue>(error: ValidationErrors) -> FieldError<S> {
    let mut errors: Vec<Value<S>> = vec![];

    error.errors().iter().for_each(|(field, kind)| match kind {
        validator::ValidationErrorsKind::Struct(e) => {
            for (_, error) in e.0.iter() {
                if let validator::ValidationErrorsKind::Field(field_errors) = error {
                    for error in field_errors {
                        let mut obj = Object::with_capacity(2);
                        obj.add_field("path", Value::scalar(field.to_string()));
                        obj.add_field(
                            "message",
                            Value::scalar(error.message.clone().unwrap_or_default().to_string()),
                        );
                        errors.push(obj.into());
                    }
                }
            }
        }
        validator::ValidationErrorsKind::List(_) => {
            warn!("List errors are not handled");
        }
        validator::ValidationErrorsKind::Field(e) => {
            for error in e {
                let mut obj = Object::with_capacity(2);
                obj.add_field("path", Value::scalar(error.code.to_string()));
                obj.add_field(
                    "message",
                    Value::scalar(error.message.clone().unwrap_or_default().to_string()),
                );
                errors.push(obj.into());
            }
        }
    });

    let mut error = Object::with_capacity(1);
    error.add_field("errors", Value::list(errors));

    let mut ext = Object::with_capacity(1);
    ext.add_field("validation-errors", error.into());

    FieldError::new("Invalid input parameters", ext.into())
}

#[derive(Clone, Copy, Debug)]
pub struct Subscription;

#[graphql_subscription]
impl Subscription {
    async fn create_thread_and_run(
        ctx: &Context,
        input: CreateThreadAndRunInput,
    ) -> Result<ThreadRunStream> {
        let user = check_user_allow_auth_token(ctx).await?;
        input.validate()?;

        let thread = ctx.locator.thread();

        let thread_id = thread.create(&user.id, &input.thread).await?;

        thread
            .create_run(
                &user.policy,
                &thread_id,
                &input.options,
                input.thread.user_message.attachments.as_ref(),
                true,
                true,
            )
            .await
    }

    async fn create_thread_run(
        ctx: &Context,
        input: CreateThreadRunInput,
    ) -> Result<ThreadRunStream> {
        let user = check_user_allow_auth_token(ctx).await?;
        input.validate()?;

        let svc = ctx.locator.thread();
        let Some(thread) = svc.get(&input.thread_id).await? else {
            return Err(CoreError::NotFound("Thread not found"));
        };

        if thread.user_id != user.id {
            return Err(CoreError::Forbidden(
                "You must be the thread owner to create a run",
            ));
        }

        svc.append_user_message(&input.thread_id, &input.additional_user_message)
            .await?;

        svc.create_run(
            &user.policy,
            &input.thread_id,
            &input.options,
            input.additional_user_message.attachments.as_ref(),
            true,
            false,
        )
        .await
    }
}

pub type Schema = RootNode<'static, Query, Mutation, Subscription>;

pub fn create_schema() -> Schema {
    Schema::new(Query, Mutation, Subscription)
}
