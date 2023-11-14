pub mod api;

mod schema;
pub use schema::create_schema;
use websocket::WebSocketTransport;

mod server;
mod ui;
mod websocket;
mod worker;

use std::{net::SocketAddr, sync::Arc};

use api::{Hub, HubError, Worker, WorkerKind};
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
use server::ServerContext;
use tarpc::server::{BaseChannel, Channel};

pub async fn attach_webserver(router: Router) -> Router {
    let ws = Arc::new(ServerContext::default());
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
        .route("/hub", routing::get(ws_handler).with_state(ws.clone()))
        .fallback(ui::handler)
        .layer(from_fn_with_state(ws, distributed_tabby_layer))
}

async fn distributed_tabby_layer(
    State(ws): State<Arc<ServerContext>>,
    request: Request<Body>,
    next: Next<Body>,
) -> axum::response::Response {
    ws.dispatch_request(request, next).await
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerContext>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(state, socket, addr))
}

async fn handle_socket(state: Arc<ServerContext>, socket: WebSocket, addr: SocketAddr) {
    let transport = WebSocketTransport::from(socket);
    let server = BaseChannel::with_defaults(transport);
    let imp = Arc::new(HubImpl::new(state.clone(), addr));
    tokio::spawn(server.execute(imp.serve())).await.unwrap()
}

pub struct HubImpl {
    ws: Arc<ServerContext>,
    conn: SocketAddr,
}

impl HubImpl {
    pub fn new(ws: Arc<ServerContext>, conn: SocketAddr) -> Self {
        Self { ws, conn }
    }
}

#[tarpc::server]
impl Hub for Arc<HubImpl> {
    async fn register_worker(
        self,
        _context: tarpc::context::Context,
        kind: WorkerKind,
        port: i32,
        name: String,
        device: String,
        arch: String,
        cpu_info: String,
        cpu_count: i32,
        cuda_devices: Vec<String>,
    ) -> Result<Worker, HubError> {
        let worker = Worker {
            name,
            kind,
            addr: format!("http://{}:{}", self.conn.ip(), port),
            device,
            arch,
            cpu_info,
            cpu_count,
            cuda_devices,
        };
        self.ws.register_worker(worker).await
    }
}
