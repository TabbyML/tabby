mod metrics;

use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use axum::{http::HeaderName, routing, Router};
use axum_extra::headers::Header;
use axum_prometheus::PrometheusMetricLayer;
use axum_tracing_opentelemetry::{middleware::OtelAxumLayer, };
use tabby_common::constants::USER_HEADER_FIELD_NAME;
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::fatal;

pub async fn run_app(api: Router, ui: Option<Router>, host: IpAddr, port: u16) {
    let (prometheus_layer, prometheus_handle) = PrometheusMetricLayer::pair();
    let app = api
        .layer(CorsLayer::permissive())
        .layer(OtelAxumLayer::default())
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
    let listener = tokio::net::TcpListener::bind(address).await.unwrap();

    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .await
        .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}

#[derive(Debug)]
pub(crate) struct MaybeUser(pub Option<String>);

pub(crate) static USER_HEADER: HeaderName = HeaderName::from_static(USER_HEADER_FIELD_NAME);

impl Header for MaybeUser {
    fn name() -> &'static axum::http::HeaderName {
        &USER_HEADER
    }

    fn decode<'i, I>(values: &mut I) -> Result<Self, axum_extra::headers::Error>
    where
        Self: Sized,
        I: Iterator<Item = &'i axum::http::HeaderValue>,
    {
        let Some(value) = values.next() else {
            return Ok(MaybeUser(None));
        };
        let str = value.to_str().expect("User email is always a valid string");
        Ok(MaybeUser(Some(str.to_string())))
    }

    fn encode<E: Extend<axum::http::HeaderValue>>(&self, _values: &mut E) {
        todo!()
    }
}

mod chat;
mod completions;
mod events;
mod health;
mod search;
mod server_setting;

pub use chat::*;
pub use completions::*;
pub use events::*;
pub use health::*;
pub use search::*;
pub use server_setting::*;
