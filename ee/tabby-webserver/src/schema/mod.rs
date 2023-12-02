pub mod auth;
pub mod worker;

use std::sync::Arc;

use auth::AuthenticationService;
use juniper::{
    graphql_object, graphql_value, EmptySubscription, FieldError, FieldResult, IntoFieldError,
    Object, RootNode, ScalarValue, Value,
};
use juniper_axum::FromAuth;
use tabby_common::api::{code::CodeSearch, event::RawEventLogger};
use validator::ValidationError;

use self::{
    auth::{validate_jwt, Invitation, RegisterError, TokenAuthError},
    worker::WorkerService,
};
use crate::schema::{
    auth::{RegisterResponse, TokenAuthResponse, VerifyTokenResponse},
    worker::Worker,
};

pub trait ServiceLocator: Send + Sync {
    fn auth(&self) -> &dyn AuthenticationService;
    fn worker(&self) -> &dyn WorkerService;
    fn code(&self) -> &dyn CodeSearch;
    fn logger(&self) -> &dyn RawEventLogger;
}

pub struct Context {
    claims: Option<auth::Claims>,
    locator: Arc<dyn ServiceLocator>,
}

impl FromAuth<Arc<dyn ServiceLocator>> for Context {
    fn build(locator: Arc<dyn ServiceLocator>, bearer: Option<String>) -> Self {
        let claims = bearer.and_then(|token| validate_jwt(&token).ok());
        Self { claims, locator }
    }
}

// To make our context usable by Juniper, we have to implement a marker trait.
impl juniper::Context for Context {}

#[derive(Default)]
pub struct Query;

#[graphql_object(context = Context)]
impl Query {
    async fn workers(ctx: &Context) -> Vec<Worker> {
        ctx.locator.worker().list_workers().await
    }

    async fn registration_token(ctx: &Context) -> FieldResult<String> {
        let token = ctx.locator.worker().read_registration_token().await?;
        Ok(token)
    }

    async fn is_admin_initialized(ctx: &Context) -> FieldResult<bool> {
        Ok(ctx.locator.auth().is_admin_initialized().await?)
    }

    async fn invitations(ctx: &Context) -> FieldResult<Vec<Invitation>> {
        if let Some(claims) = &ctx.claims {
            if claims.user_info().is_admin() {
                return Ok(ctx.locator.auth().list_invitations().await?);
            }
        }
        Err(unauthorized("Only admin is able to query invitations"))
    }
}

#[derive(Default)]
pub struct Mutation;

#[graphql_object(context = Context)]
impl Mutation {
    async fn reset_registration_token(ctx: &Context) -> FieldResult<String> {
        if let Some(claims) = &ctx.claims {
            if claims.user_info().is_admin() {
                let reg_token = ctx.locator.worker().reset_registration_token().await?;
                return Ok(reg_token);
            }
        }
        Err(unauthorized(
            "Only admin is able to reset registration token",
        ))
    }

    async fn register(
        ctx: &Context,
        email: String,
        password1: String,
        password2: String,
        invitation_code: Option<String>,
    ) -> Result<RegisterResponse, RegisterError> {
        ctx
            .locator
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

    async fn verify_token(ctx: &Context, token: String) -> FieldResult<VerifyTokenResponse> {
        Ok(ctx.locator.auth().verify_token(token).await?)
    }

    async fn create_invitation(ctx: &Context, email: String) -> FieldResult<i32> {
        if let Some(claims) = &ctx.claims {
            if claims.user_info().is_admin() {
                return Ok(ctx.locator.auth().create_invitation(email).await?);
            }
        }
        Err(unauthorized("Only admin is able to create invitation"))
    }

    async fn delete_invitation(ctx: &Context, id: i32) -> FieldResult<i32> {
        if let Some(claims) = &ctx.claims {
            if claims.user_info().is_admin() {
                return Ok(ctx.locator.auth().delete_invitation(id).await?);
            }
        }
        Err(unauthorized("Only admin is able to delete invitation"))
    }
}

struct ValidationErrors(Vec<ValidationError>);

impl<S: ScalarValue> IntoFieldError<S> for ValidationErrors {
    fn into_field_error(self) -> FieldError<S> {
        let errors = self
            .0
            .into_iter()
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
        let mut ext = Object::with_capacity(2);
        ext.add_field("code", Value::scalar("validation-error".to_string()));
        ext.add_field("errors", Value::list(errors));

        FieldError::new("Invalid input parameters", ext.into())
    }
}

pub type Schema = RootNode<'static, Query, Mutation, EmptySubscription<Context>>;

pub fn create_schema() -> Schema {
    Schema::new(Query, Mutation, EmptySubscription::new())
}

fn unauthorized(msg: &str) -> FieldError {
    FieldError::new(msg, graphql_value!("Unauthorized"))
}
