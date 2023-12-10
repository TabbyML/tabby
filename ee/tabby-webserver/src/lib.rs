pub mod api;

mod schema;
use api::Hub;
pub use schema::create_schema;
use tabby_common::api::{
    code::{CodeSearch, SearchResponse},
    event::RawEventLogger,
};
use tokio::sync::Mutex;
use tracing::{error, warn};
use websocket::WebSocketTransport;

mod repositories;
mod service;
mod ui;
mod websocket;

use std::{net::SocketAddr, sync::Arc};

use axum::{
    extract::{ws::WebSocket, ConnectInfo, State, WebSocketUpgrade},
    http::Request,
    middleware::{from_fn_with_state, Next},
    response::IntoResponse,
    routing, Extension, Router,
};
use hyper::Body;
use juniper_axum::{graphiql, graphql, playground};
pub use schema::create_schema;
use schema::{
    worker::{RegisterWorkerError, Worker, WorkerKind},
    Schema, ServiceLocator,
};
use server::ServerContext;
use service::create_service_locator;
use tabby_common::api::{
    accelerator::Accelerator,
    code::{CodeSearch, SearchResponse},
    event::RawEventLogger,
};
use tarpc::server::{BaseChannel, Channel};
use tokio::sync::Mutex;
use tracing::{error, warn};
use websocket::WebSocketTransport;

pub mod api;

mod db;
mod repositories;
mod schema;
mod server;
mod ui;
mod websocket;

pub async fn attach_webserver(
    api: Router,
    ui: Router,
    logger: Arc<dyn RawEventLogger>,
    code: Arc<dyn CodeSearch>,
) -> (Router, Router) {
    let ctx = create_service_locator(logger, code).await;
    let schema = Arc::new(create_schema());

    let api = api
        .layer(from_fn_with_state(ctx.clone(), distributed_tabby_layer))
        .route(
            "/graphql",
            routing::post(graphql::<Arc<Schema>, Arc<dyn ServiceLocator>>).with_state(ctx.clone()),
        )
        .route("/graphql", routing::get(playground("/graphql", None)))
        .layer(Extension(schema))
        .route("/hub", routing::get(ws_handler).with_state(ctx))
        .nest("/repositories", repositories::routes());

    let ui = ui
        .route("/graphiql", routing::get(graphiql("/graphql", None)))
        .fallback(ui::handler);

    (api, ui)
}

async fn distributed_tabby_layer(
    State(ws): State<Arc<dyn ServiceLocator>>,
    request: Request<Body>,
    next: Next<Body>,
) -> axum::response::Response {
    ws.worker().dispatch_request(request, next).await
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<dyn ServiceLocator>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(state, socket, addr))
}

async fn handle_socket(state: Arc<dyn ServiceLocator>, socket: WebSocket, addr: SocketAddr) {
    let transport = WebSocketTransport::from(socket);
    let server = BaseChannel::with_defaults(transport);
    let imp = Arc::new(HubImpl::new(state.clone(), addr));
    tokio::spawn(server.execute(imp.serve())).await.unwrap()
}

pub struct HubImpl {
    ctx: Arc<dyn ServiceLocator>,
    conn: SocketAddr,

    worker_addr: Arc<Mutex<String>>,
}

impl HubImpl {
    pub fn new(ctx: Arc<dyn ServiceLocator>, conn: SocketAddr) -> Self {
        Self {
            ctx,
            conn,
            worker_addr: Arc::new(Mutex::new("".to_owned())),
        }
    }
}

impl Drop for HubImpl {
    fn drop(&mut self) {
        let ctx = self.ctx.clone();
        let worker_addr = self.worker_addr.clone();

        tokio::spawn(async move {
            let worker_addr = worker_addr.lock().await;
            if !worker_addr.is_empty() {
                ctx.worker().unregister_worker(worker_addr.as_str()).await;
            }
        });
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
        accelerators: Vec<Accelerator>,
        token: String,
    ) -> Result<Worker, RegisterWorkerError> {
        if token.is_empty() {
            return Err(RegisterWorkerError::InvalidToken(
                "Empty worker token".to_string(),
            ));
        }
        let server_token = match self.ctx.worker().read_registration_token().await {
            Ok(t) => t,
            Err(err) => {
                error!("fetch server token: {}", err.to_string());
                return Err(RegisterWorkerError::InvalidToken(
                    "Failed to fetch server token".to_string(),
                ));
            }
        };
        if server_token != token {
            return Err(RegisterWorkerError::InvalidToken(
                "Token mismatch".to_string(),
            ));
        }

        let mut worker_addr = self.worker_addr.lock().await;
        if !worker_addr.is_empty() {
            return Err(RegisterWorkerError::RegisterWorkerOnce);
        }

        let addr = format!("http://{}:{}", self.conn.ip(), port);
        *worker_addr = addr.clone();

        let worker = Worker {
            name,
            kind,
            addr,
            device,
            arch,
            cpu_info,
            cpu_count,
            accelerators,
        };
        self.ctx.worker().register_worker(worker).await
    }

    async fn log_event(self, _context: tarpc::context::Context, content: String) {
        self.ctx.logger().log(content)
    }

    async fn search(
        self,
        _context: tarpc::context::Context,
        q: String,
        limit: usize,
        offset: usize,
    ) -> SearchResponse {
        match self.ctx.code().search(&q, limit, offset).await {
            Ok(serp) => serp,
            Err(err) => {
                warn!("Failed to search: {}", err);
                SearchResponse::default()
            }
        }
    }

    async fn search_in_language(
        self,
        _context: tarpc::context::Context,
        language: String,
        tokens: Vec<String>,
        limit: usize,
        offset: usize,
    ) -> SearchResponse {
        match self
            .ctx
            .code()
            .search_in_language(&language, &tokens, limit, offset)
            .await
        {
            Ok(serp) => serp,
            Err(err) => {
                warn!("Failed to search: {}", err);
                SearchResponse::default()
            }
        }
    }
}
