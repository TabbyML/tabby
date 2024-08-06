pub mod extract;
pub mod websocket;

use axum::{
    extract::{Extension, State, WebSocketUpgrade},
    response::Response,
};
use extract::{extract_bearer_token, AuthBearer};
use futures::future;
use juniper::{Variables};
use juniper_axum::{
    extract::JuniperRequest,
    response::JuniperResponse,
    subscriptions::{self},
};
use juniper_graphql_ws::{ConnectionConfig, Schema};


pub trait FromAuth<S> {
    fn build(state: S, bearer: Option<String>) -> Self;
}

#[cfg_attr(text, axum::debug_handler)]
pub async fn graphql<S, C>(
    State(state): State<C>,
    Extension(schema): Extension<S>,
    AuthBearer(bearer): AuthBearer,
    JuniperRequest(req): JuniperRequest<S::ScalarValue>,
) -> JuniperResponse<S::ScalarValue>
where
    S: Schema, // TODO: Refactor in the way we don't depend on `juniper_graphql_ws::Schema` here.
    S::Context: FromAuth<C>,
{
    let ctx = S::Context::build(state, bearer);
    JuniperResponse(req.execute(schema.root_node(), &ctx).await)
}

pub async fn subscriptions<S, C>(
    Extension(schema): Extension<S>,
    State(state): State<C>,
    ws: WebSocketUpgrade,
) -> Response
where
    C: Send + Sync + Unpin + 'static,
    S: Schema,
    S::Context: FromAuth<C>,
{
    ws.protocols(["graphql-transport-ws", "graphql-ws"])
        .on_upgrade(move |socket| {
            let init = move |params: Variables<S::ScalarValue>| -> future::Ready<
                Result<ConnectionConfig<S::Context>, tabby_schema::CoreError>,
            > {
                // Extract authorization header from connection init payload
                let bearer = params
                    .get("authorization")
                    .and_then(|v| v.as_string_value())
                    .and_then(extract_bearer_token);
                let ctx = S::Context::build(state, bearer);
                future::ready(Ok(ConnectionConfig::new(ctx)))
            };
            subscriptions::serve_ws(socket, schema, init)
        })
}
