mod metrics;

use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use axum::{routing, Router};
use axum_prometheus::PrometheusMetricLayer;
use axum_tracing_opentelemetry::opentelemetry_tracing_layer;
use hyper::Server;
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::fatal;

pub async fn run_app(api: Router, ui: Option<Router>, host: IpAddr, port: u16) {
    let (prometheus_layer, prometheus_handle) = PrometheusMetricLayer::pair();
    let app = api
        .layer(CorsLayer::permissive())
        .layer(opentelemetry_tracing_layer())
        .layer(prometheus_layer)
        .route(
            "/metrics",
            routing::get(metrics::metrics).with_state(Arc::new(prometheus_handle)),
        );

    let app = if let Some(ui) = ui {
        app.merge(ui)
    } else {
        app
    };

    let address = SocketAddr::from((host, port));
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
mod search;

pub use chat::*;
pub use completions::*;
pub use events::*;
pub use health::*;
pub use search::*;
