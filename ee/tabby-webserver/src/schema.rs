use juniper::{graphql_object, EmptyMutation, EmptySubscription, RootNode};

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

pub type Schema =
    RootNode<'static, Query, EmptyMutation<ServerContext>, EmptySubscription<ServerContext>>;

pub fn create_schema() -> Schema {
    Schema::new(Query, EmptyMutation::new(), EmptySubscription::new())
}
