use std::{net::SocketAddr, str::FromStr};

use juniper::{
    graphql_object, EmptyMutation, EmptySubscription, FieldResult, GraphQLEnum, GraphQLObject,
    RootNode,
};
use tokio::sync::Mutex;

use crate::registry::CodeSearchWorkerRegistry;

#[derive(Default)]
pub struct Context {
    code: Mutex<CodeSearchWorkerRegistry>,
}

// To make our context usable by Juniper, we have to implement a marker trait.
impl juniper::Context for Context {}

#[derive(GraphQLEnum)]
enum WorkerKind {
    CodeSearch,
}

#[derive(GraphQLObject)]
struct Worker {
    kind: WorkerKind,
    address: String,
}

#[derive(Default)]
pub struct Query;

#[graphql_object(context = Context)]
impl Query {
    fn workers() -> Vec<Worker> {
        vec![]
    }
}

pub struct Mutation;

#[graphql_object(context = Context)]
impl Mutation {
    async fn register_code_search_worker(context: &Context, server_addr: String) -> bool {
        let addr = SocketAddr::from_str(&server_addr).unwrap();
        context.code.lock().await.register(addr).await.is_ok()
    }
}

pub type Schema = RootNode<'static, Query, Mutation, EmptySubscription<Context>>;

pub fn new() -> Schema {
    Schema::new(Query, Mutation, EmptySubscription::new())
}
