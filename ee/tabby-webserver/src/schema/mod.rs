pub mod auth;

use std::sync::Arc;

use auth::AuthenticationService;
use juniper::{
    graphql_object, graphql_value, EmptySubscription, FieldError, FieldResult, IntoFieldError,
    Object, RootNode, ScalarValue, Value,
};
use juniper_axum::FromAuth;
use validator::ValidationError;

use self::auth::validate_jwt;
use crate::{
    api::Worker,
    schema::auth::{RegisterResponse, TokenAuthResponse, VerifyTokenResponse},
    server::ServerContext,
};

pub struct Context {
    claims: Option<auth::Claims>,
    server: Arc<ServerContext>,
}

impl FromAuth<Arc<ServerContext>> for Context {
    fn build(server: Arc<ServerContext>, bearer: Option<String>) -> Self {
        let claims = bearer.and_then(|token| validate_jwt(&token).ok());
        Self { claims, server }
    }
}

// To make our context usable by Juniper, we have to implement a marker trait.
impl juniper::Context for Context {}

#[derive(Default)]
pub struct Query;

#[graphql_object(context = Context)]
impl Query {
    async fn workers(ctx: &Context) -> Vec<Worker> {
        ctx.server.list_workers().await
    }

    async fn registration_token(ctx: &Context) -> FieldResult<String> {
        let token = ctx.server.read_registration_token().await?;
        Ok(token)
    }
}

#[derive(Default)]
pub struct Mutation;

#[graphql_object(context = Context)]
impl Mutation {
    async fn reset_registration_token(ctx: &Context) -> FieldResult<String> {
        if let Some(claims) = &ctx.claims {
            if claims.user_info().is_admin() {
                let reg_token = ctx.server.reset_registration_token().await?;
                return Ok(reg_token);
            }
        }
        Err(FieldError::new(
            "Only admin is able to reset registration token",
            graphql_value!("Unauthorized"),
        ))
    }

    async fn register(
        ctx: &Context,
        email: String,
        password1: String,
        password2: String,
    ) -> FieldResult<RegisterResponse> {
        ctx.server
            .auth()
            .register(email, password1, password2)
            .await
    }

    async fn token_auth(
        ctx: &Context,
        email: String,
        password: String,
    ) -> FieldResult<TokenAuthResponse> {
        ctx.server.auth().token_auth(email, password).await
    }

    async fn verify_token(ctx: &Context, token: String) -> FieldResult<VerifyTokenResponse> {
        ctx.server.auth().verify_token(token).await
    }
}

#[derive(Debug)]
pub struct ValidationErrors {
    pub errors: Vec<ValidationError>,
}

impl<S: ScalarValue> IntoFieldError<S> for ValidationErrors {
    fn into_field_error(self) -> FieldError<S> {
        let errors = self
            .errors
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
