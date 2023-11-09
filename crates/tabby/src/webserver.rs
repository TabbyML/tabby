use std::net::{Ipv4Addr, SocketAddr};

use axum::Server;
use clap::Args;
use tracing::info;

use crate::fatal;

#[derive(Args)]
pub struct WebserverArgs {}

pub async fn main(_args: &WebserverArgs) {
    let app = tabby_webserver::api_router();
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8080));
    info!("Listening at {}", address);

    Server::bind(&address)
        .serve(app.into_make_service())
        .await
        .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}
