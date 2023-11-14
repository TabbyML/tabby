use juniper::{graphql_object, EmptyMutation, EmptySubscription, RootNode};

use crate::{api::Worker, webserver::Webserver};

// To make our context usable by Juniper, we have to implement a marker trait.
impl juniper::Context for Webserver {}

#[derive(Default)]
pub struct Query;

#[graphql_object(context = Webserver)]
impl Query {
    async fn workers(ctx: &Webserver) -> Vec<Worker> {
        ctx.list_workers().await
    }
}

pub type Schema = RootNode<'static, Query, EmptyMutation<Webserver>, EmptySubscription<Webserver>>;

pub fn create_schema() -> Schema {
    Schema::new(Query, EmptyMutation::new(), EmptySubscription::new())
}
