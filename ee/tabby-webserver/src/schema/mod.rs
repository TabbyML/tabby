pub mod auth;
pub mod job;
pub mod worker;

use std::sync::Arc;

use auth::{
    validate_jwt, AuthenticationService, Invitation, InvitationNext, RefreshTokenError,
    RefreshTokenResponse, RegisterError, RegisterResponse, TokenAuthError, TokenAuthResponse, User,
    VerifyTokenResponse,
};
use job::{JobRun, JobService};
use juniper::{
    graphql_object, graphql_value, EmptySubscription, FieldError, FieldResult, IntoFieldError,
    Object, RootNode, ScalarValue, Value, ID,
};
use juniper_axum::{relay, FromAuth};
use tabby_common::api::{code::CodeSearch, event::RawEventLogger};
use tracing::error;
use validator::ValidationErrors;
use worker::{Worker, WorkerService};

pub trait ServiceLocator: Send + Sync {
    fn auth(&self) -> &dyn AuthenticationService;
    fn worker(&self) -> &dyn WorkerService;
    fn code(&self) -> &dyn CodeSearch;
    fn logger(&self) -> &dyn RawEventLogger;
    fn job(&self) -> &dyn JobService;
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

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl<S: ScalarValue> IntoFieldError<S> for CoreError {
    fn into_field_error(self) -> FieldError<S> {
        match self {
            Self::Unauthorized(msg) => FieldError::new(msg, graphql_value!("Unauthorized")),
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
            "Only admin is able to read registeration_token",
        ))
    }

    async fn is_admin_initialized(ctx: &Context) -> Result<bool> {
        Ok(ctx.locator.auth().is_admin_initialized().await?)
    }

    #[deprecated]
    async fn invitations(ctx: &Context) -> Result<Vec<Invitation>> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                return Ok(ctx
                    .locator
                    .auth()
                    .list_invitations(None, None, None, None)
                    .await?
                    .into_iter()
                    .map(|x| x.into())
                    .collect());
            }
        }
        Err(CoreError::Unauthorized(
            "Only admin is able to query invitations",
        ))
    }

    async fn me(ctx: &Context) -> Result<User> {
        if let Some(claims) = &ctx.claims {
            let user = ctx.locator.auth().get_user_by_email(&claims.sub).await?;
            Ok(user)
        } else {
            Err(CoreError::Unauthorized("Not logged in"))
        }
    }

    #[deprecated]
    async fn users(ctx: &Context) -> Result<Vec<User>> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                return Ok(ctx
                    .locator
                    .auth()
                    .list_users(None, None, None, None)
                    .await?);
            }
        }
        Err(CoreError::Unauthorized("Only admin is able to query users"))
    }

    async fn users_next(
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> FieldResult<relay::Connection<User>> {
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

    async fn invitations_next(
        ctx: &Context,
        after: Option<String>,
        before: Option<String>,
        first: Option<i32>,
        last: Option<i32>,
    ) -> FieldResult<relay::Connection<InvitationNext>> {
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
    ) -> FieldResult<relay::Connection<JobRun>> {
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
                            .job()
                            .list_job_runs(after, before, first, last)
                            .await
                        {
                            Ok(job_runs) => Ok(job_runs),
                            Err(err) => Err(FieldError::from(err)),
                        }
                    },
                )
                .await;
            }
        }
        Err(FieldError::from(CoreError::Unauthorized(
            "Only admin is able to query job runs",
        )))
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
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                return Ok(ctx.locator.auth().create_invitation(email).await?);
            }
        }
        Err(CoreError::Unauthorized(
            "Only admin is able to create invitation",
        ))
    }

    #[deprecated]
    async fn delete_invitation(ctx: &Context, id: i32) -> Result<i32> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                let id = ctx
                    .locator
                    .auth()
                    .delete_invitation(ID::new(id.to_string()))
                    .await?;
                return Ok(id.parse::<i32>().unwrap());
            }
        }
        Err(CoreError::Unauthorized(
            "Only admin is able to delete invitation",
        ))
    }

    async fn delete_invitation_next(ctx: &Context, id: ID) -> Result<ID> {
        if let Some(claims) = &ctx.claims {
            if claims.is_admin {
                return Ok(ctx.locator.auth().delete_invitation(id).await?);
            }
        }
        Err(CoreError::Unauthorized(
            "Only admin is able to delete invitation",
        ))
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
