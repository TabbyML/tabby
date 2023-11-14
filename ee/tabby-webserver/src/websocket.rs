use std::{
    marker::PhantomData,
    pin::Pin,
    task::{Context, Poll},
};

use axum::extract::ws;
use futures::{Sink, Stream};
use pin_project::pin_project;
use tokio::net::TcpStream;
use tokio_tungstenite as tt;
use tokio_tungstenite::tungstenite as ts;

pub trait IntoData {
    fn into_data(self) -> Option<Vec<u8>>;
}

impl IntoData for ws::Message {
    fn into_data(self) -> Option<Vec<u8>> {
        match self {
            ws::Message::Binary(x) => Some(x),
            _ => None,
        }
    }
}

impl IntoData for ts::Message {
    fn into_data(self) -> Option<Vec<u8>> {
        match self {
            ts::Message::Binary(x) => Some(x),
            _ => None,
        }
    }
}

#[pin_project]
pub struct WebSocketTransport<Req, Resp, Message, Transport, Error>
where
    Message: IntoData + From<Vec<u8>>,
    Transport: Stream<Item = Result<Message, Error>> + Sink<Message, Error = Error>,
{
    #[pin]
    inner: Transport,
    ghost: PhantomData<(Req, Resp)>,
}

impl<Req, Resp> From<ws::WebSocket>
    for WebSocketTransport<Req, Resp, ws::Message, ws::WebSocket, axum::Error>
{
    fn from(inner: ws::WebSocket) -> Self {
        Self {
            inner,
            ghost: PhantomData,
        }
    }
}

impl<Req, Resp> From<tt::WebSocketStream<tt::MaybeTlsStream<TcpStream>>>
    for WebSocketTransport<
        Req,
        Resp,
        ts::Message,
        tt::WebSocketStream<tt::MaybeTlsStream<TcpStream>>,
        tt::tungstenite::Error,
    >
{
    fn from(inner: tokio_tungstenite::WebSocketStream<tt::MaybeTlsStream<TcpStream>>) -> Self {
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
