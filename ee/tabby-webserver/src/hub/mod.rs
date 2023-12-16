pub mod api;
mod websocket;

use std::{net::SocketAddr, sync::Arc};

use api::{Hub, RegisterWorkerRequest};
use axum::{
    extract::{ws::WebSocket, ConnectInfo, State, WebSocketUpgrade},
    response::IntoResponse,
    TypedHeader,
};
use hyper::{Body, StatusCode};
use juniper_axum::extract::AuthBearer;
use tabby_common::api::code::SearchResponse;
use tarpc::server::{BaseChannel, Channel};
use tracing::warn;
use websocket::WebSocketTransport;

use crate::schema::{worker::Worker, ServiceLocator};

pub(crate) async fn ws_handler(
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

struct HubImpl {
    ctx: Arc<dyn ServiceLocator>,
    worker_addr: String,
}

impl HubImpl {
    fn new(ctx: Arc<dyn ServiceLocator>, worker_addr: String) -> Self {
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
