use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use axum::{routing, Router};
use axum_prometheus::PrometheusMetricLayer;
use axum_tracing_opentelemetry::opentelemetry_tracing_layer;
use hyper::Server;
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::fatal;

pub async fn run_app(app: Router, port: u16) {
    let (prometheus_layer, prometheus_handle) = PrometheusMetricLayer::pair();
    let app = app
        .route(
            "/v1/metrics",
            routing::get(metrics::metrics).with_state(Arc::new(prometheus_handle)),
        )
        .layer(CorsLayer::permissive())
        .layer(opentelemetry_tracing_layer())
        .layer(prometheus_layer);

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
    info!("Listening at {}", address);

    Server::bind(&address)
        .serve(app.into_make_service_with_connect_info::<SocketAddr>())
        .await
        .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}

mod chat;
mod completions;
mod events;
mod health;
mod metrics;
mod search;

pub use chat::*;
pub use completions::*;
pub use events::*;
pub use health::*;
pub use metrics::*;
pub use search::*;
