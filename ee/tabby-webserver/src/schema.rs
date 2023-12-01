pub mod auth;

use std::sync::Arc;

use hyper::Server;
use juniper::{
    graphql_object, graphql_value, EmptySubscription, FieldError, FieldResult, RootNode,
};
use juniper_axum::FromAuth;

use crate::{
    api::Worker,
    schema::auth::{RegisterResponse, TokenAuthResponse, VerifyTokenResponse},
    server::{
        auth::{AuthenticationService, RegisterInput, TokenAuthInput, validate_jwt},
        ServerContext,
    },
};

pub struct Context {
    claims: Option<auth::Claims>,
    server: Arc<ServerContext>,
}

impl FromAuth<Arc<ServerContext>> for Context {
    fn build(server: Arc<ServerContext>, bearer: Option<String>) -> Self {
        let claims = bearer.map(|token| validate_jwt(&token).ok()).flatten();
        Self {
            claims,
            server,
        }
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
    async fn reset_registration_token(
        ctx: &Context,
    ) -> FieldResult<String> {
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
        let input = RegisterInput {
            email,
            password1,
            password2,
        };
        ctx.server.auth().register(input).await
    }

    async fn token_auth(
        ctx: &Context,
        email: String,
        password: String,
    ) -> FieldResult<TokenAuthResponse> {
        let input = TokenAuthInput { email, password };
        ctx.server.auth().token_auth(input).await
    }

    async fn verify_token(ctx: &Context, token: String) -> FieldResult<VerifyTokenResponse> {
        ctx.server.auth().verify_token(token).await
    }
}

pub type Schema = RootNode<'static, Query, Mutation, EmptySubscription<Context>>;

pub fn create_schema() -> Schema {
    Schema::new(Query, Mutation, EmptySubscription::new())
}
