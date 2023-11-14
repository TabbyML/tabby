mod api;
pub mod schema;
mod ui;
mod webserver;
mod worker;

use std::{
    net::SocketAddr,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use api::WebserverApi;
use axum::{
    async_trait,
    extract::{
        ws::{Message, WebSocket},
        ConnectInfo, State, WebSocketUpgrade,
    },
    http::Request,
    middleware::{from_fn_with_state, Next},
    response::IntoResponse,
    routing, Error, Extension, Router,
};
use futures::{stream, Sink, Stream, StreamExt, TryStreamExt};
use hyper::Body;
use juniper::EmptySubscription;
use juniper_axum::{graphiql, graphql, playground};
use schema::{Mutation, Query, Schema};
use tarpc::{
    server::{BaseChannel, Channel},
    transport::{self, channel::unbounded},
    ClientMessage,
};
use tokio::io::{AsyncRead, ReadBuf};
use webserver::Webserver;

pub fn attach_webserver(router: Router) -> Router {
    let ws = Arc::new(Webserver::default());
    let schema = Arc::new(Schema::new(Query, Mutation, EmptySubscription::new()));

    let app = Router::new()
        .route("/graphql", routing::get(playground("/graphql", None)))
        .route("/graphiql", routing::get(graphiql("/graphql", None)))
        .route(
            "/graphql",
            routing::post(graphql::<Arc<Schema>, Arc<Webserver>>).with_state(ws.clone()),
        )
        .route("/ws", routing::get(ws_handler).with_state(ws.clone()))
        .layer(Extension(schema));

    router
        .merge(app)
        .fallback(ui::handler)
        .layer(from_fn_with_state(ws, distributed_tabby_layer))
}

async fn distributed_tabby_layer(
    State(ws): State<Arc<Webserver>>,
    request: Request<Body>,
    next: Next<Body>,
) -> axum::response::Response {
    ws.dispatch_request(request, next).await
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<Webserver>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    println!("{addr} connected.");
    // finalize the upgrade process by returning upgrade callback.
    // we can customize the callback by sending additional info such as address.
    // ws.on_upgrade(move |socket| handle_socket(state, socket, addr))
}

async fn handle_socket(state: Arc<Webserver>, mut socket: WebSocket, who: SocketAddr) {
    let x = socket.map_ok(|x| serde_json::from_slice(&x.into_data()).unwrap());

    let x: Box<dyn Stream<Item = Result<String, axum::Error>>> = Box::new(x);

    //let (client, server) = unbounded();
    // let server = BaseChannel::with_defaults(WebSocketTransport(socket));
    // tokio::spawn(server.execute(state.serve())).await;
}
