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
use tracing::{error, warn};
use validator::ValidationErrors;
use worker::{Worker, WorkerService};

use self::{
    email::{EmailService, EmailSetting},
    repository::{RepositoryError, RepositoryService},
    setting::{ServerSetting, SettingService},
};
use crate::schema::{
    auth::{JWTPayload, OAuthCredential, OAuthProvider},
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

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl<S: ScalarValue> IntoFieldError<S> for CoreError {
    fn into_field_error(self) -> FieldError<S> {
        match self {
            Self::Unauthorized(msg) => {
                FieldError::new(msg, graphql_value!({"code": "UNAUTHORIZED"}))
            }
            _ => self.into(),
        }
    }
}

// To make our context usable by Juniper, we have to implement a marker trait.
impl juniper::Context for Context {}

#[derive(Default)]
pub struct Query;

#[graphql_object(context = Context)]
impl Query {
    async fn workers(ctx: &Context) -> Result<Vec<Worker>> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                let workers = ctx.locator.worker().list_workers().await;
                return Ok(workers);
            }
        }
        Err(CoreError::Unauthorized(
            "Only admin is able to read workers",
        ))
    }

    async fn registration_token(ctx: &Context) -> Result<String> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                let token = ctx.locator.worker().read_registration_token().await?;
                return Ok(token);
            }
        }
        Err(CoreError::Unauthorized(
            "Only admin is able to read registration_token",
        ))
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
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
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
        }
        Err(FieldError::from(CoreError::Unauthorized(
            "Only admin is able to query users",
        )))
    }

    async fn invitations(
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> FieldResult<Connection<Invitation>> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                return relay::query_async(
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
                .await;
            }
        }
        Err(FieldError::from(CoreError::Unauthorized(
            "Only admin is able to query users",
        )))
    }

    async fn job_runs(
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> FieldResult<Connection<JobRun>> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                return relay::query_async(
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
                .await;
            }
        }
        Err(FieldError::from(CoreError::Unauthorized(
            "Only admin is able to query job runs",
        )))
    }

    async fn email_setting(ctx: &Context) -> Result<Option<EmailSetting>> {
        let Some(JWTPayload { is_admin: true, .. }) = &ctx.claims else {
            return Err(CoreError::Unauthorized(
                "Only admin can access server settings",
            ));
        };
        let val = ctx.locator.email().get_email_setting().await?;
        Ok(val)
    }

    async fn server_setting(ctx: &Context) -> Result<ServerSetting> {
        let Some(JWTPayload { is_admin: true, .. }) = &ctx.claims else {
            return Err(CoreError::Unauthorized(
                "Only admin can access server settings",
            ));
        };
        let val = ctx.locator.setting().read_server_setting().await?;
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
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                let Some(credentials) = ctx.locator.auth().read_oauth_credential(provider).await?
                else {
                    return Ok(None);
                };
                return Ok(Some(credentials));
            }
        }
        Err(CoreError::Unauthorized(
            "Only admin is able to query oauth credential",
        ))
    }
}

#[derive(Default)]
pub struct Mutation;

#[graphql_object(context = Context)]
impl Mutation {
    async fn reset_registration_token(ctx: &Context) -> Result<String> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                let reg_token = ctx.locator.worker().reset_registration_token().await?;
                return Ok(reg_token);
            }
        }
        Err(CoreError::Unauthorized(
            "Only admin is able to reset registration token",
        ))
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
        ctx.locator.auth().update_user_active(&id, active).await?;
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
        let Some(JWTPayload { is_admin: true, .. }) = &ctx.claims else {
            return Err(CoreError::Unauthorized(
                "Only admin is able to create invitation",
            ));
        };
        let invitation = ctx.locator.auth().create_invitation(email.clone()).await?;
        let email_sent = ctx
            .locator
            .email()
            .send_invitation_email(email, invitation.code)
            .await;
        if let Err(e) = email_sent {
            warn!(
                "Failed to send invitation email, please check your SMTP settings are correct: {e}"
            );
        }
        Ok(invitation.id)
    }

    async fn create_repository(
        ctx: &Context,
        name: String,
        git_url: String,
    ) -> Result<ID, RepositoryError> {
        ctx.locator
            .repository()
            .create_repository(name, git_url)
            .await
    }

    async fn delete_repository(ctx: &Context, id: ID) -> Result<bool> {
        Ok(ctx.locator.repository().delete_repository(&id).await?)
    }

    async fn update_repository(
        ctx: &Context,
        id: ID,
        name: String,
        git_url: String,
    ) -> Result<bool> {
        Ok(ctx
            .locator
            .repository()
            .update_repository(&id, name, git_url)
            .await?)
    }

    async fn delete_invitation(ctx: &Context, id: ID) -> Result<ID> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                return Ok(ctx.locator.auth().delete_invitation(&id).await?);
            }
        }
        Err(CoreError::Unauthorized(
            "Only admin is able to delete invitation",
        ))
    }

    async fn update_oauth_credential(
        ctx: &Context,
        provider: OAuthProvider,
        client_id: String,
        client_secret: String,
        redirect_uri: Option<String>,
    ) -> Result<bool> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                ctx.locator
                    .auth()
                    .update_oauth_credential(provider, client_id, client_secret, redirect_uri)
                    .await?;
                return Ok(true);
            }
        }
        Err(CoreError::Unauthorized(
            "Only admin is able to update oauth credential",
        ))
    }

    async fn delete_oauth_credential(ctx: &Context, provider: OAuthProvider) -> Result<bool> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                ctx.locator.auth().delete_oauth_credential(provider).await?;
                return Ok(true);
            }
        }
        Err(CoreError::Unauthorized(
            "Only admin is able to delete oauth credential",
        ))
    }

    async fn update_email_setting(
        ctx: &Context,
        smtp_username: String,
        smtp_password: Option<String>,
        smtp_server: String,
    ) -> Result<bool> {
        let Some(JWTPayload { is_admin: true, .. }) = &ctx.claims else {
            return Err(CoreError::Unauthorized(
                "Only admin can access server settings",
            ));
        };
        ctx.locator
            .email()
            .update_email_setting(smtp_username, smtp_password, smtp_server)
            .await?;
        Ok(true)
    }

    async fn update_server_setting(
        ctx: &Context,
        security_allowed_register_domain_list: Vec<String>,
        security_disable_client_side_telemetry: bool,
        network_external_url: String,
    ) -> Result<bool> {
        let Some(JWTPayload { is_admin: true, .. }) = &ctx.claims else {
            return Err(CoreError::Unauthorized(
                "Only admin can access server settings",
            ));
        };
        ctx.locator
            .setting()
            .update_server_setting(ServerSetting {
                security_allowed_register_domain_list,
                security_disable_client_side_telemetry,
                network_external_url,
            })
            .await?;
        Ok(true)
    }

    async fn delete_email_setting(ctx: &Context) -> Result<bool> {
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
