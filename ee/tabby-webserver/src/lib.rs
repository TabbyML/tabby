pub mod api;

mod schema;
use api::Hub;
pub use schema::create_schema;
use serde::{Deserialize, Serialize};
use tabby_common::api::{
    code::{CodeSearch, SearchResponse},
    event::RawEventLogger,
};
use tracing::warn;
use websocket::WebSocketTransport;

mod repositories;
mod service;
mod ui;
mod websocket;

use std::{net::SocketAddr, sync::Arc};

use axum::{
    extract::{ws::WebSocket, ConnectInfo, State, WebSocketUpgrade},
    headers::Header,
    http::{HeaderName, Request},
    middleware::{from_fn_with_state, Next},
    response::IntoResponse,
    routing, Extension, Router, TypedHeader,
};
use hyper::{Body, StatusCode};
use juniper_axum::{extract::AuthBearer, graphiql, graphql, playground};
use schema::{
    worker::{Worker, WorkerKind},
    Schema, ServiceLocator,
};
use service::create_service_locator;
use tarpc::server::{BaseChannel, Channel};

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
        .route("/hub", routing::get(ws_handler).with_state(ctx.clone()))
        .nest("/repositories", repositories::routes(ctx.clone()));

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

#[derive(Serialize, Deserialize)]
struct RegisterWorkerRequest {
    kind: WorkerKind,
    port: i32,
    name: String,
    device: String,
    arch: String,
    cpu_info: String,
    cpu_count: i32,
    cuda_devices: Vec<String>,
}

pub static REGISTER_WORKER_HEADER: HeaderName = HeaderName::from_static("x-tabby-register-worker");

impl Header for RegisterWorkerRequest {
    fn name() -> &'static axum::http::HeaderName {
        &REGISTER_WORKER_HEADER
    }

    fn decode<'i, I>(values: &mut I) -> Result<Self, axum::headers::Error>
    where
        Self: Sized,
        I: Iterator<Item = &'i axum::http::HeaderValue>,
    {
        let mut x: Vec<_> = values
            .map(|x| serde_json::from_slice(x.as_bytes()))
            .collect();
        if let Some(x) = x.pop() {
            x.map_err(|_| axum::headers::Error::invalid())
        } else {
            Err(axum::headers::Error::invalid())
        }
    }

    fn encode<E: Extend<axum::http::HeaderValue>>(&self, _values: &mut E) {
        todo!()
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<dyn ServiceLocator>>,
    AuthBearer(token): AuthBearer,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    TypedHeader(request): TypedHeader<RegisterWorkerRequest>,
) -> impl IntoResponse {
    let unauthorized = axum::response::Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .body(Body::empty())
        .unwrap()
        .into_response();

    let Some(token) = token else {
        return unauthorized;
    };

    let Ok(registeration_token) = state.worker().read_registration_token().await else {
        return unauthorized;
    };

    if token != registeration_token {
        return unauthorized;
    }

    let addr = format!("http://{}:{}", addr.ip(), request.port);

    let worker = Worker {
        name: request.name,
        kind: request.kind,
        addr,
        device: request.device,
        arch: request.arch,
        cpu_info: request.cpu_info,
        cpu_count: request.cpu_count,
        cuda_devices: request.cuda_devices,
    };

    ws.on_upgrade(move |socket| handle_socket(state, socket, worker))
        .into_response()
}

async fn handle_socket(state: Arc<dyn ServiceLocator>, socket: WebSocket, worker: Worker) {
    let transport = WebSocketTransport::from(socket);
    let server = BaseChannel::with_defaults(transport);
    let imp = Arc::new(HubImpl::new(state.clone(), worker.addr.clone()));
    state.worker().register_worker(worker).await.unwrap();
    tokio::spawn(server.execute(imp.serve())).await.unwrap()
}

pub struct HubImpl {
    ctx: Arc<dyn ServiceLocator>,
    worker_addr: String,
}

impl HubImpl {
    pub fn new(ctx: Arc<dyn ServiceLocator>, worker_addr: String) -> Self {
        Self { ctx, worker_addr }
    }
}

impl Drop for HubImpl {
    fn drop(&mut self) {
        let ctx = self.ctx.clone();
        let worker_addr = self.worker_addr.clone();

        tokio::spawn(async move {
            ctx.worker().unregister_worker(worker_addr.as_str()).await;
        });
    }
}

#[tarpc::server]
impl Hub for Arc<HubImpl> {
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
