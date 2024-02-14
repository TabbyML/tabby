pub mod auth;
pub mod email;
pub mod job;
pub mod repository;
pub mod setting;
pub mod worker;

use std::sync::Arc;

use auth::{
    validate_jwt, AuthenticationService, Invitation, RefreshTokenError, RefreshTokenResponse,
    RegisterError, RegisterResponse, TokenAuthError, TokenAuthResponse, User, VerifyTokenResponse,
};
use job::{JobRun, JobService};
use juniper::{
    graphql_object, graphql_value, EmptySubscription, FieldError, FieldResult, IntoFieldError,
    Object, RootNode, ScalarValue, Value, ID,
};
use juniper_axum::{
    relay::{self, Connection},
    FromAuth,
};
use tabby_common::api::{code::CodeSearch, event::RawEventLogger};
use tracing::error;
use validator::{Validate, ValidationErrors};
use worker::{Worker, WorkerService};

use self::{
    auth::{PasswordResetInput, RequestPasswordResetInput, UpdateOAuthCredentialInput},
    email::{EmailService, EmailSetting, EmailSettingInput},
    repository::RepositoryService,
    setting::{
        NetworkSetting, NetworkSettingInput, SecuritySetting, SecuritySettingInput, SettingService,
    },
};
use crate::schema::{
    auth::{JWTPayload, OAuthCredential, OAuthProvider, RequestInvitationInput},
    repository::Repository,
};

pub trait ServiceLocator: Send + Sync {
    fn auth(&self) -> Arc<dyn AuthenticationService>;
    fn worker(&self) -> Arc<dyn WorkerService>;
    fn code(&self) -> Arc<dyn CodeSearch>;
    fn logger(&self) -> Arc<dyn RawEventLogger>;
    fn job(&self) -> Arc<dyn JobService>;
    fn repository(&self) -> Arc<dyn RepositoryService>;
    fn email(&self) -> Arc<dyn EmailService>;
    fn setting(&self) -> Arc<dyn SettingService>;
}

pub struct Context {
    claims: Option<auth::JWTPayload>,
    locator: Arc<dyn ServiceLocator>,
}

impl FromAuth<Arc<dyn ServiceLocator>> for Context {
    fn build(locator: Arc<dyn ServiceLocator>, bearer: Option<String>) -> Self {
        let claims = bearer.and_then(|token| validate_jwt(&token).ok());
        Self { claims, locator }
    }
}

type Result<T, E = CoreError> = std::result::Result<T, E>;

#[derive(thiserror::Error, Debug)]
pub enum CoreError {
    #[error("{0}")]
    Unauthorized(&'static str),

    #[error("Invalid ID Error")]
    InvalidIDError,

    #[error("Invalid input parameters")]
    InvalidInput(#[from] ValidationErrors),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl<S: ScalarValue> IntoFieldError<S> for CoreError {
    fn into_field_error(self) -> FieldError<S> {
        match self {
            Self::Unauthorized(msg) => {
                FieldError::new(msg, graphql_value!({"code": "UNAUTHORIZED"}))
            }
            Self::InvalidInput(errors) => from_validation_errors(errors),
            _ => self.into(),
        }
    }
}

// To make our context usable by Juniper, we have to implement a marker trait.
impl juniper::Context for Context {}

fn check_admin(ctx: &Context) -> Result<(), CoreError> {
    if let Some(JWTPayload { is_admin: true, .. }) = &ctx.claims {
        Ok(())
    } else {
        Err(CoreError::Unauthorized("You must be admin to do that"))
    }
}

#[derive(Default)]
pub struct Query;

#[graphql_object(context = Context)]
impl Query {
    async fn workers(ctx: &Context) -> Result<Vec<Worker>> {
        check_admin(ctx)?;
        let workers = ctx.locator.worker().list_workers().await;
        return Ok(workers);
    }

    async fn registration_token(ctx: &Context) -> Result<String> {
        check_admin(ctx)?;
        let token = ctx.locator.worker().read_registration_token().await?;
        return Ok(token);
    }

    async fn is_admin_initialized(ctx: &Context) -> Result<bool> {
        Ok(ctx.locator.auth().is_admin_initialized().await?)
    }

    async fn me(ctx: &Context) -> Result<User> {
        if let Some(claims) = &ctx.claims {
            let user = ctx.locator.auth().get_user_by_email(&claims.sub).await?;
            Ok(user)
        } else {
            Err(CoreError::Unauthorized("Not logged in"))
        }
    }

    async fn users(
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> FieldResult<Connection<User>> {
        check_admin(ctx)?;
        return relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                match ctx
                    .locator
                    .auth()
                    .list_users(after, before, first, last)
                    .await
                {
                    Ok(users) => Ok(users),
                    Err(err) => Err(FieldError::from(err)),
                }
            },
        )
        .await;
    }

    async fn invitations(
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> FieldResult<Connection<Invitation>> {
        check_admin(ctx)?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                match ctx
                    .locator
                    .auth()
                    .list_invitations(after, before, first, last)
                    .await
                {
                    Ok(invitations) => Ok(invitations),
                    Err(err) => Err(FieldError::from(err)),
                }
            },
        )
        .await
    }

    async fn job_runs(
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> FieldResult<Connection<JobRun>> {
        check_admin(ctx)?;
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                Ok(ctx
                    .locator
                    .job()
                    .list_job_runs(after, before, first, last)
                    .await?)
            },
        )
        .await
    }

    async fn email_setting(ctx: &Context) -> Result<Option<EmailSetting>> {
        check_admin(ctx)?;
        let val = ctx.locator.email().read_email_setting().await?;
        Ok(val)
    }

    async fn network_setting(ctx: &Context) -> Result<NetworkSetting> {
        check_admin(ctx)?;
        let val = ctx.locator.setting().read_network_setting().await?;
        Ok(val)
    }

    async fn security_setting(ctx: &Context) -> Result<SecuritySetting> {
        check_admin(ctx)?;
        let val = ctx.locator.setting().read_security_setting().await?;
        Ok(val)
    }

    async fn repositories(
        &self,
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> FieldResult<Connection<Repository>> {
        relay::query_async(
            after,
            before,
            first,
            last,
            |after, before, first, last| async move {
                Ok(ctx
                    .locator
                    .repository()
                    .list_repositories(after, before, first, last)
                    .await?)
            },
        )
        .await
    }

    async fn oauth_credential(
        ctx: &Context,
        provider: OAuthProvider,
    ) -> Result<Option<OAuthCredential>> {
        check_admin(ctx)?;
        let Some(credentials) = ctx.locator.auth().read_oauth_credential(provider).await? else {
            return Ok(None);
        };
        Ok(Some(credentials))
    }

    async fn oauth_callback_url(ctx: &Context, provider: OAuthProvider) -> Result<String> {
        check_admin(ctx)?;
        Ok(ctx.locator.auth().oauth_callback_url(provider).await?)
    }
}

#[derive(Default)]
pub struct Mutation;

#[graphql_object(context = Context)]
impl Mutation {
    async fn reset_registration_token(ctx: &Context) -> Result<String> {
        check_admin(ctx)?;
        let reg_token = ctx.locator.worker().reset_registration_token().await?;
        Ok(reg_token)
    }

    async fn request_invitation_email(
        ctx: &Context,
        input: RequestInvitationInput,
    ) -> Result<Invitation> {
        input.validate()?;
        Ok(ctx.locator.auth().request_invitation(input).await?)
    }

    async fn request_password_reset_email(
        ctx: &Context,
        input: RequestPasswordResetInput,
    ) -> Result<bool> {
        input.validate()?;
        ctx.locator
            .auth()
            .request_password_reset(input.email)
            .await?;
        Ok(true)
    }

    async fn password_reset(ctx: &Context, input: PasswordResetInput) -> Result<bool> {
        input.validate()?;
        ctx.locator
            .auth()
            .password_reset(&input.email, &input.code, &input.password)
            .await
            .map_err(anyhow::Error::from)?;
        Ok(true)
    }

    async fn reset_user_auth_token(ctx: &Context) -> Result<bool> {
        if let Some(claims) = &ctx.claims {
            ctx.locator
                .auth()
                .reset_user_auth_token(&claims.sub)
                .await?;
            Ok(true)
        } else {
            Err(CoreError::Unauthorized("You're not logged in"))
        }
    }

    async fn update_user_active(ctx: &Context, id: ID, active: bool) -> Result<bool> {
        check_admin(ctx)?;
        ctx.locator.auth().update_user_active(&id, active).await?;
        Ok(true)
    }

    async fn update_user_role(ctx: &Context, id: ID, is_admin: bool) -> Result<bool> {
        check_admin(ctx)?;
        ctx.locator.auth().update_user_role(&id, is_admin).await?;
        Ok(true)
    }

    async fn register(
        ctx: &Context,
        email: String,
        password1: String,
        password2: String,
        invitation_code: Option<String>,
    ) -> Result<RegisterResponse, RegisterError> {
        ctx.locator
            .auth()
            .register(email, password1, password2, invitation_code)
            .await
    }

    async fn token_auth(
        ctx: &Context,
        email: String,
        password: String,
    ) -> Result<TokenAuthResponse, TokenAuthError> {
        ctx.locator.auth().token_auth(email, password).await
    }

    async fn verify_token(ctx: &Context, token: String) -> Result<VerifyTokenResponse> {
        Ok(ctx.locator.auth().verify_access_token(&token).await?)
    }

    async fn refresh_token(
        ctx: &Context,
        refresh_token: String,
    ) -> Result<RefreshTokenResponse, RefreshTokenError> {
        ctx.locator.auth().refresh_token(refresh_token).await
    }

    async fn create_invitation(ctx: &Context, email: String) -> Result<ID> {
        check_admin(ctx)?;
        let invitation = ctx.locator.auth().create_invitation(email.clone()).await?;
        Ok(invitation.id)
    }

    async fn send_test_email(ctx: &Context, to: String) -> Result<bool> {
        ctx.locator
            .email()
            .send_test_email(to)
            .await
            .map_err(anyhow::Error::from)?;
        Ok(true)
    }

    async fn create_repository(ctx: &Context, name: String, git_url: String) -> Result<ID> {
        check_admin(ctx)?;
        Ok(ctx
            .locator
            .repository()
            .create_repository(name, git_url)
            .await
            .map_err(anyhow::Error::from)?)
    }

    async fn delete_repository(ctx: &Context, id: ID) -> Result<bool> {
        check_admin(ctx)?;
        Ok(ctx.locator.repository().delete_repository(&id).await?)
    }

    async fn update_repository(
        ctx: &Context,
        id: ID,
        name: String,
        git_url: String,
    ) -> Result<bool> {
        check_admin(ctx)?;
        Ok(ctx
            .locator
            .repository()
            .update_repository(&id, name, git_url)
            .await?)
    }

    async fn delete_invitation(ctx: &Context, id: ID) -> Result<ID> {
        check_admin(ctx)?;
        Ok(ctx.locator.auth().delete_invitation(&id).await?)
    }

    async fn update_oauth_credential(
        ctx: &Context,
        input: UpdateOAuthCredentialInput,
    ) -> Result<bool> {
        check_admin(ctx)?;
        input.validate()?;
        ctx.locator.auth().update_oauth_credential(input).await?;
        Ok(true)
    }

    async fn delete_oauth_credential(ctx: &Context, provider: OAuthProvider) -> Result<bool> {
        check_admin(ctx)?;
        ctx.locator.auth().delete_oauth_credential(provider).await?;
        Ok(true)
    }

    async fn update_email_setting(ctx: &Context, input: EmailSettingInput) -> Result<bool> {
        check_admin(ctx)?;
        input.validate()?;
        ctx.locator.email().update_email_setting(input).await?;
        Ok(true)
    }

    async fn update_security_setting(ctx: &Context, input: SecuritySettingInput) -> Result<bool> {
        check_admin(ctx)?;
        input.validate()?;
        ctx.locator.setting().update_security_setting(input).await?;
        Ok(true)
    }

    async fn update_network_setting(ctx: &Context, input: NetworkSettingInput) -> Result<bool> {
        check_admin(ctx)?;
        input.validate()?;
        ctx.locator.setting().update_network_setting(input).await?;
        Ok(true)
    }

    async fn delete_email_setting(ctx: &Context) -> Result<bool> {
        check_admin(ctx)?;
        ctx.locator.email().delete_email_setting().await?;
        Ok(true)
    }
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
