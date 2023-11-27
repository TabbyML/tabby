use juniper::{graphql_object, EmptySubscription, FieldResult, RootNode};

use crate::{api::Worker, server::ServerContext};

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
    async fn reset_registration_token(ctx: &ServerContext) -> FieldResult<String> {
        let token = ctx.reset_registration_token().await?;
        Ok(token)
    }
}

pub type Schema = RootNode<'static, Query, Mutation, EmptySubscription<ServerContext>>;

pub fn create_schema() -> Schema {
    Schema::new(Query, Mutation, EmptySubscription::new())
}
