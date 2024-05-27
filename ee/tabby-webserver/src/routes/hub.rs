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
use tabby_common::api::event::LogEntry;
use tabby_schema::ServiceLocator;
use tarpc::server::{BaseChannel, Channel};

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
    _addr: IpAddr,
    _req: ConnectHubRequest,
) {
    let transport = WebSocketTransport::from(socket);
    let server = BaseChannel::with_defaults(transport);
    let imp = Arc::new(HubImpl::new(state.locator.clone()));
    tokio::spawn(server.execute(imp.serve())).await.unwrap()
}

struct HubImpl {
    ctx: Arc<dyn ServiceLocator>,
}

impl HubImpl {
    fn new(ctx: Arc<dyn ServiceLocator>) -> Self {
        Self { ctx }
    }
}

#[tarpc::server]
impl Hub for Arc<HubImpl> {
    async fn write_log(self, _context: tarpc::context::Context, x: LogEntry) {
        self.ctx.logger().write(x)
    }
}
