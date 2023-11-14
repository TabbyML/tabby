use std::{net::SocketAddr, sync::Arc};

use juniper::{
    graphql_object, graphql_value, EmptySubscription, FieldError, GraphQLEnum, GraphQLObject,
    IntoFieldError, RootNode, ScalarValue, Value,
};
use juniper_axum::FromStateAndClientAddr;
use serde::{Deserialize, Serialize};

use crate::webserver::{Webserver, WebserverError};

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

#[derive(GraphQLEnum, Serialize, Deserialize, Clone, Debug)]
pub enum WorkerKind {
    Completion,
    Chat,
}

#[derive(GraphQLObject, Serialize, Deserialize, Clone, Debug)]
pub struct Worker {
    pub kind: WorkerKind,
    pub name: String,
    pub addr: String,
    pub device: String,
    pub arch: String,
    pub cpu_info: String,
    pub cpu_count: i32,
    pub cuda_devices: Vec<String>,
}

#[derive(Default)]
pub struct Query;

#[graphql_object(context = Request)]
impl Query {
    async fn workers(request: &Request) -> Vec<Worker> {
        request.ws.list_workers().await
    }
}

pub struct Mutation;

#[graphql_object(context = Request)]
impl Mutation {
    async fn register_worker(
        request: &Request,
        port: i32,
        kind: WorkerKind,
        name: String,
        device: String,
        arch: String,
        cpu_info: String,
        cpu_count: i32,
        cuda_devices: Vec<String>,
    ) -> Result<Worker, WebserverError> {
        let ws = &request.ws;
        let worker = Worker {
            name,
            kind,
            addr: format!("http://{}:{}", request.client_addr.ip(), port),
            device,
            arch,
            cpu_info,
            cpu_count,
            cuda_devices,
        };
        ws.register_worker(worker).await
    }
}

pub type Schema = RootNode<'static, Query, Mutation, EmptySubscription<Request>>;

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

#[tarpc::service]
pub trait WebserverApi {
    async fn register_worker_as(
        kind: WorkerKind,
        port: i32,
        name: String,
        device: String,
        arch: String,
        cpu_info: String,
        cpu_count: i32,
        cuda_devices: Vec<String>,
    ) -> Worker;
}
