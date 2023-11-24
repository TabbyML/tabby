pub mod auth;

use juniper::{
    graphql_object, graphql_value, EmptySubscription, FieldError, FieldResult, RootNode,
};

use crate::{
    api::Worker,
    schema::auth::{RegisterResponse, TokenAuthResponse, VerifyTokenResponse},
    server::{
        auth::{validate_jwt, AuthenticationService, RegisterInput, TokenAuthInput},
        ServerContext,
    },
};

// To make our context usable by Juniper, we have to implement a marker trait.
impl juniper::Context for ServerContext {}

#[derive(Default)]
pub struct Query;

#[graphql_object(context = ServerContext)]
impl Query {
    async fn workers(ctx: &ServerContext) -> Vec<Worker> {
        ctx.list_workers().await
    }

    async fn registration_token(ctx: &ServerContext) -> FieldResult<String> {
        let token = ctx.read_registration_token().await?;
        Ok(token)
    }
}

#[derive(Default)]
pub struct Mutation;

#[graphql_object(context = ServerContext)]
impl Mutation {
    async fn reset_registration_token(
        ctx: &ServerContext,
        token: Option<String>,
    ) -> FieldResult<String> {
        if let Some(Ok(claims)) = token.map(|t| validate_jwt(&t)) {
            if claims.user_info().is_admin() {
                let reg_token = ctx.reset_registration_token().await?;
                return Ok(reg_token);
            }
        }
        Err(FieldError::new(
            "Only admin is able to reset registration token",
            graphql_value!("Unauthorized"),
        ))
    }

    async fn register(
        ctx: &ServerContext,
        email: String,
        password1: String,
        password2: String,
    ) -> FieldResult<RegisterResponse> {
        let input = RegisterInput {
            email,
            password1,
            password2,
        };
        let resp = ctx.auth().register(input).await?;
        Ok(resp)
    }

    async fn token_auth(
        ctx: &ServerContext,
        email: String,
        password: String,
    ) -> FieldResult<TokenAuthResponse> {
        let input = TokenAuthInput { email, password };
        let resp = ctx.auth().token_auth(input).await?;
        Ok(resp)
    }

    async fn verify_token(ctx: &ServerContext, token: String) -> FieldResult<VerifyTokenResponse> {
        let resp = ctx.auth().verify_token(token).await?;
        Ok(resp)
    }
}

pub type Schema = RootNode<'static, Query, Mutation, EmptySubscription<ServerContext>>;

pub fn create_schema() -> Schema {
    Schema::new(Query, Mutation, EmptySubscription::new())
}
