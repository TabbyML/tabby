pub mod api;

mod schema;
pub use schema::create_schema;
use websocket::WebSocketTransport;
use tracing::{debug, error};

mod server;
mod ui;
mod websocket;
mod db;

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
    let conn = Arc::new(db::init_db().await.unwrap());
    let ctx = Arc::new(ServerContext::new(conn));
    let schema = Arc::new(create_schema());

    let app = Router::new()
        .route("/graphql", routing::get(playground("/graphql", None)))
        .route("/graphiql", routing::get(graphiql("/graphql", None)))
        .route(
            "/graphql",
            routing::post(graphql::<Arc<Schema>>).with_state(ctx.clone()),
        )
        .layer(Extension(schema));

    router
        .merge(app)
        .route("/hub", routing::get(ws_handler).with_state(ctx.clone()))
        .fallback(ui::handler)
        .layer(from_fn_with_state(ctx, distributed_tabby_layer))
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
    ctx: Arc<ServerContext>,
    conn: SocketAddr,
}

impl HubImpl {
    pub fn new(ctx: Arc<ServerContext>, conn: SocketAddr) -> Self {
        Self { ctx, conn }
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
        token: String,
    ) -> Result<Worker, HubError> {
        if token.is_empty() {
            return Err(HubError::InvalidToken("Empty worker token".to_string()));
        }
        let server_token = match self.ctx.token().await {
            Ok(t) => t,
            Err(err) => {
                error!("fetch server token: {}", err.to_string());
                return Err(HubError::InvalidToken("Failed to fetch server token".to_string()));
            }
        };
        if server_token != token {
            debug!("server_token=`{}`, worker_token=`{}`", server_token, token);
            return Err(HubError::InvalidToken("Token mismatch".to_string()));
        }

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
        self.ctx.register_worker(worker).await
    }
}
