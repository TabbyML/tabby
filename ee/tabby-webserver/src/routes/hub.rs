use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use axum::{
    body::Body,
    extract::{ws::WebSocket, ConnectInfo, State, WebSocketUpgrade},
    http::StatusCode,
    response::IntoResponse,
};
use axum_extra::TypedHeader;
use tabby_common::api::{code::SearchResponse, event::LogEntry};
use tabby_schema::ServiceLocator;
use tarpc::server::{BaseChannel, Channel};
use tracing::warn;

use crate::{
    axum::{extract::AuthBearer, websocket::WebSocketTransport},
    hub::{ConnectHubRequest, Hub},
};

pub(crate) struct HubState {
    locator: Arc<dyn ServiceLocator>,
}

impl HubState {
    pub fn new(locator: Arc<dyn ServiceLocator>) -> Self {
        HubState { locator }
    }
}

pub(crate) async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<HubState>>,
    AuthBearer(token): AuthBearer,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    TypedHeader(request): TypedHeader<ConnectHubRequest>,
) -> impl IntoResponse {
    let unauthorized = axum::response::Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .body(Body::empty())
        .unwrap()
        .into_response();

    let Some(token) = token else {
        return unauthorized;
    };

    let Ok(registeration_token) = state.locator.worker().read_registration_token().await else {
        return unauthorized;
    };

    if token != registeration_token {
        return unauthorized;
    }

    ws.on_upgrade(move |socket| handle_socket(state, socket, addr.ip(), request))
        .into_response()
}

async fn handle_socket(
    state: Arc<HubState>,
    socket: WebSocket,
    addr: IpAddr,
    req: ConnectHubRequest,
) {
    let transport = WebSocketTransport::from(socket);
    let server = BaseChannel::with_defaults(transport);
    let addr = match req {
        ConnectHubRequest::Worker(worker) => {
            let worker = worker.create_worker(addr);
            let addr = worker.addr.clone();
            match state.locator.worker().register(worker).await {
                Ok(_) => Some(addr),
                Err(err) => {
                    warn!("Failed to register worker: {}", err);
                    return;
                }
            }
        }
    };
    let imp = Arc::new(HubImpl::new(state.locator.clone(), addr));
    tokio::spawn(server.execute(imp.serve())).await.unwrap()
}

struct HubImpl {
    ctx: Arc<dyn ServiceLocator>,
    worker_addr: Option<String>,
}

impl HubImpl {
    fn new(ctx: Arc<dyn ServiceLocator>, worker_addr: Option<String>) -> Self {
        Self { ctx, worker_addr }
    }
}

impl Drop for HubImpl {
    fn drop(&mut self) {
        let ctx = self.ctx.clone();
        if let Some(worker_addr) = self.worker_addr.clone() {
            tokio::spawn(async move {
                ctx.worker().unregister(worker_addr.as_str()).await;
            });
        }
    }
}

#[tarpc::server]
impl Hub for Arc<HubImpl> {
    async fn write_log(self, _context: tarpc::context::Context, x: LogEntry) {
        self.ctx.logger().write(x)
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
        git_url: String,
        language: String,
        tokens: Vec<String>,
        limit: usize,
        offset: usize,
    ) -> SearchResponse {
        match self
            .ctx
            .code()
            .search_in_language(&git_url, &language, &tokens, limit, offset)
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
