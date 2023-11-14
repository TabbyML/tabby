use juniper::{
    graphql_object, graphql_value, EmptyMutation, EmptySubscription, FieldError, IntoFieldError,
    RootNode, ScalarValue, Value,
};

use crate::{
    api::Worker,
    webserver::{Webserver, WebserverError},
};

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

pub fn create_schema() -> Schema {
    Schema::new(Query, EmptyMutation::new(), EmptySubscription::new())
}
