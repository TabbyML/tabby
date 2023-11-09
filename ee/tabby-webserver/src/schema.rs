use std::{net::SocketAddr, sync::Arc};

use juniper::{
    graphql_object, graphql_value, EmptySubscription, FieldError, FieldResult, GraphQLEnum,
    RootNode, Value,
};
use tracing::info;

use crate::{juniper_axum::FromStateAndClientAddr, Webserver};

pub struct Request {
    ws: Arc<Webserver>,
    client_addr: SocketAddr,
}

impl FromStateAndClientAddr<Request, Arc<Webserver>> for Request {
    fn build(ws: Arc<Webserver>, client_addr: SocketAddr) -> Request {
        Request { ws, client_addr }
    }
}

// To make our context usable by Juniper, we have to implement a marker trait.
impl juniper::Context for Request {}

#[derive(GraphQLEnum, Debug)]
enum WorkerKind {
    Completion,
    Chat,
}

#[derive(Default)]
pub struct Query;

#[graphql_object(context = Request)]
impl Query {
    async fn workers(request: &Request, kind: WorkerKind) -> Vec<String> {
        match kind {
            WorkerKind::Completion => request.ws.completion.list().await,
            WorkerKind::Chat => request.ws.chat.list().await,
        }
    }
}

pub struct Mutation;

#[graphql_object(context = Request)]
impl Mutation {
    async fn register_worker(
        request: &Request,
        kind: WorkerKind,
        port: i32,
    ) -> FieldResult<String> {
        let addr = SocketAddr::new(request.client_addr.ip(), port as u16);
        let addr = match kind {
            WorkerKind::Completion => request.ws.completion.register(addr).await,
            WorkerKind::Chat => request.ws.chat.register(addr).await,
        };

        if let Some(addr) = addr {
            info!("registering <{:?}> worker running at {}", kind, addr);
            Ok(addr)
        } else {
            Err(FieldError::new("Failed to register worker", Value::Null))
        }
    }
}

pub type Schema = RootNode<'static, Query, Mutation, EmptySubscription<Request>>;

pub fn new() -> Schema {
    Schema::new(Query, Mutation, EmptySubscription::new())
}
