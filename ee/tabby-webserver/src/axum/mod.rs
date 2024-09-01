pub mod extract;
pub mod websocket;

use async_trait::async_trait;
use axum::{
    extract::{Extension, State, WebSocketUpgrade},
    response::Response,
};
use extract::{extract_bearer_token, AuthBearer};
use futures::FutureExt;
use juniper::{BoxFuture, Variables};
use juniper_axum::{
    extract::JuniperRequest,
    response::JuniperResponse,
    subscriptions::{self},
};
use juniper_graphql_ws::{ConnectionConfig, Schema};

#[async_trait]
pub trait FromAuth<S> {
    async fn build(state: S, token: Option<String>) -> Self;
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
    let ctx = S::Context::build(state, bearer).await;
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
            let init = move |params: Variables<S::ScalarValue>| -> BoxFuture<
                Result<ConnectionConfig<S::Context>, tabby_schema::CoreError>,
            > {
                // Extract authorization header from connection init payload
                let bearer = params
                    .get("authorization")
                    .and_then(|v| v.as_string_value())
                    .and_then(extract_bearer_token);

                // Allow auth token for websocket connection
                let ctx = S::Context::build(state, bearer);
                ctx.then(|ctx| async move { Ok(ConnectionConfig::new(ctx)) })
                    .boxed()
            };
            subscriptions::serve_ws(socket, schema, init)
        })
}
