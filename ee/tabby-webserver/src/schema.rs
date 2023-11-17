use juniper::{graphql_object, EmptySubscription, RootNode};

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
}

#[derive(Default)]
pub struct Mutation;

#[graphql_object(context = ServerContext)]
impl Mutation {
    async fn reset_registration_token(ctx: &ServerContext) -> String {
        match ctx.reset_registration_token().await {
            Ok(token) => token,
            Err(err) => format!("error: {}", err),
        }
    }
}

pub type Schema = RootNode<'static, Query, Mutation, EmptySubscription<ServerContext>>;

pub fn create_schema() -> Schema {
    Schema::new(Query, Mutation, EmptySubscription::new())
}
