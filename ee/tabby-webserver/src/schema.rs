use std::{net::SocketAddr, sync::Arc};

use juniper::{
    graphql_object, graphql_value, EmptySubscription, FieldError, GraphQLEnum, IntoFieldError,
    RootNode, ScalarValue, Value,
};
use juniper_axum::FromStateAndClientAddr;

use crate::{Webserver, WebserverError};

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
pub enum WorkerKind {
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
        token: String,
        kind: WorkerKind,
        port: i32,
    ) -> Result<String, WebserverError> {
        let ws = &request.ws;
        ws.register_worker(token, request.client_addr, kind, port)
            .await
    }
}

pub type Schema = RootNode<'static, Query, Mutation, EmptySubscription<Request>>;

pub fn new() -> Schema {
    Schema::new(Query, Mutation, EmptySubscription::new())
}

impl<S: ScalarValue> IntoFieldError<S> for WebserverError {
    fn into_field_error(self) -> FieldError<S> {
        let msg = format!("{}", &self);
        match self {
            WebserverError::InvalidToken(token) => FieldError::new(
                msg,
                graphql_value!({
                    "token": token
                }),
            ),
            _ => FieldError::new(msg, Value::Null),
        }
    }
}
