mod api;
pub mod schema;
mod ui;
mod webserver;
mod worker;

use std::{
    marker::PhantomData,
    net::SocketAddr,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use api::{WebserverApi, WebserverApiClient};
use axum::{
    extract::{
        ws::{Message, WebSocket},
        ConnectInfo, State, WebSocketUpgrade,
    },
    http::Request,
    middleware::{from_fn_with_state, Next},
    response::IntoResponse,
    routing, Extension, Router,
};
use futures::{Sink, Stream};
use hyper::Body;
use juniper::EmptySubscription;
use juniper_axum::{graphiql, graphql, playground};
use pin_project::pin_project;
use schema::{Mutation, Query, Schema};
use tarpc::server::{BaseChannel, Channel};
use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, MaybeTlsStream};
use webserver::{Webserver, WebserverImpl};

pub async fn attach_webserver(router: Router) -> Router {
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
    ws.on_upgrade(move |socket| handle_socket(state, socket, addr))
}

async fn handle_socket(state: Arc<Webserver>, socket: WebSocket, addr: SocketAddr) {
    let transport = WebSocketTransport::from(socket);
    let server = BaseChannel::with_defaults(transport);
    let imp = Arc::new(WebserverImpl::new(state.clone(), addr));
    tokio::spawn(server.execute(imp.serve())).await.unwrap()
}

pub fn tarpc_context() -> tarpc::context::Context {
    tarpc::context::current()
}

pub async fn create_webserver_api_client(addr: String) -> WebserverApiClient {
    let (socket, _) = connect_async(&addr).await.unwrap();
    WebserverApiClient::new(Default::default(), WebSocketTransport::from(socket)).spawn()
}

trait IntoData {
    fn into_data(self) -> Option<Vec<u8>>;
}

impl IntoData for Message {
    fn into_data(self) -> Option<Vec<u8>> {
        match self {
            Message::Binary(x) => Some(x),
            _ => None,
        }
    }
}

impl IntoData for tokio_tungstenite::tungstenite::Message {
    fn into_data(self) -> Option<Vec<u8>> {
        match self {
            tokio_tungstenite::tungstenite::Message::Binary(x) => Some(x),
            _ => None,
        }
    }
}

#[pin_project]
struct WebSocketTransport<Req, Resp, Message, Transport, Error>
where
    Message: IntoData + From<Vec<u8>>,
    Transport: Stream<Item = Result<Message, Error>> + Sink<Message, Error = Error>,
{
    #[pin]
    inner: Transport,
    ghost: PhantomData<(Req, Resp)>,
}

impl<Req, Resp> From<WebSocket>
    for WebSocketTransport<
        Req,
        Resp,
        axum::extract::ws::Message,
        axum::extract::ws::WebSocket,
        axum::Error,
    >
{
    fn from(inner: WebSocket) -> Self {
        Self {
            inner,
            ghost: PhantomData,
        }
    }
}

impl<Req, Resp> From<tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>>
    for WebSocketTransport<
        Req,
        Resp,
        tokio_tungstenite::tungstenite::Message,
        tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>,
        tokio_tungstenite::tungstenite::Error,
    >
{
    fn from(inner: tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>) -> Self {
        Self {
            inner,
            ghost: PhantomData,
        }
    }
}

impl<Req, Resp, Message, Transport, Error> Stream
    for WebSocketTransport<Req, Resp, Message, Transport, Error>
where
    Req: for<'de> serde::Deserialize<'de>,
    Message: IntoData + From<Vec<u8>> + std::fmt::Debug,
    Transport: Stream<Item = Result<Message, Error>> + Sink<Message, Error = Error>,
{
    type Item = Result<Req, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match futures::ready!(self.as_mut().project().inner.poll_next(cx)) {
            Some(Ok(msg)) => {
                let bin = msg.into_data();
                match bin {
                    Some(bin) => Poll::Ready(Some(Ok(bincode::deserialize_from::<&[u8], Req>(
                        bin.as_ref(),
                    )
                    .unwrap()))),
                    None => Poll::Ready(None),
                }
            }
            Some(Err(err)) => Poll::Ready(Some(Err(err))),
            None => Poll::Ready(None),
        }
    }
}

impl<Req, Resp, Message, Transport, Error> Sink<Resp>
    for WebSocketTransport<Req, Resp, Message, Transport, Error>
where
    Resp: serde::Serialize,
    Message: IntoData + From<Vec<u8>>,
    Transport: Stream<Item = Result<Message, Error>> + Sink<Message, Error = Error>,
{
    type Error = Error;

    fn poll_ready(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.as_mut().project().inner.poll_ready(cx)
    }

    fn start_send(mut self: Pin<&mut Self>, item: Resp) -> Result<(), Self::Error> {
        let msg = Message::from(bincode::serialize(&item).unwrap());
        self.as_mut().project().inner.start_send(msg)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.as_mut().project().inner.poll_flush(cx)
    }

    fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.as_mut().project().inner.poll_close(cx)
    }
}
