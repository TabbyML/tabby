pub mod api;

mod schema;
pub use schema::create_schema;
use websocket::WebSocketTransport;

mod ui;
mod webserver;
mod websocket;
mod worker;

use std::{net::SocketAddr, sync::Arc};

use api::WebserverApi;
use axum::{
    extract::{ws::WebSocket, ConnectInfo, State, WebSocketUpgrade},
    http::Request,
    middleware::{from_fn_with_state, Next},
    response::IntoResponse,
    routing, Extension, Router,
};
use hyper::Body;
use juniper_axum::{graphiql, graphql, playground};
use schema::Schema;
use tarpc::server::{BaseChannel, Channel};
use webserver::{Webserver, WebserverImpl};

pub async fn attach_webserver(router: Router) -> Router {
    let ws = Arc::new(Webserver::default());
    let schema = Arc::new(create_schema());

    let app = Router::new()
        .route("/graphql", routing::get(playground("/graphql", None)))
        .route("/graphiql", routing::get(graphiql("/graphql", None)))
        .route(
            "/graphql",
            routing::post(graphql::<Arc<Schema>>).with_state(ws.clone()),
        )
        .layer(Extension(schema));

    router
        .merge(app)
        .route("/ws", routing::get(ws_handler).with_state(ws.clone()))
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
    ws.on_upgrade(move |socket| handle_socket(state, socket, addr))
}

async fn handle_socket(state: Arc<Webserver>, socket: WebSocket, addr: SocketAddr) {
    let transport = WebSocketTransport::from(socket);
    let server = BaseChannel::with_defaults(transport);
    let imp = Arc::new(WebserverImpl::new(state.clone(), addr));
    tokio::spawn(server.execute(imp.serve())).await.unwrap()
}
