use juniper::{
    graphql_object, graphql_value, EmptyMutation, EmptySubscription, FieldError, GraphQLEnum,
    GraphQLObject, IntoFieldError, RootNode, ScalarValue, Value,
};
use serde::{Deserialize, Serialize};

use crate::webserver::{Webserver, WebserverError};

// To make our context usable by Juniper, we have to implement a marker trait.
impl juniper::Context for Webserver {}

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

#[graphql_object(context = Webserver)]
impl Query {
    async fn workers(ctx: &Webserver) -> Vec<Worker> {
        ctx.list_workers().await
    }
}

pub type Schema = RootNode<'static, Query, EmptyMutation<Webserver>, EmptySubscription<Webserver>>;

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
