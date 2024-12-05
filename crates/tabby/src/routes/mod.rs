mod metrics;

use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use axum::{routing, Router};
use axum_prometheus::PrometheusMetricLayer;
use tower_http::cors::CorsLayer;

use crate::fatal;

pub async fn run_app(api: Router, ui: Option<Router>, host: IpAddr, port: u16) {
    let (prometheus_layer, prometheus_handle) = PrometheusMetricLayer::pair();
    let app = api
        .layer(CorsLayer::permissive())
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
    let version = env!("CARGO_PKG_VERSION");
    println!(
        r#"
████████╗ █████╗ ██████╗ ██████╗ ██╗   ██╗
╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝
   ██║   ███████║██████╔╝██████╔╝ ╚████╔╝ 
   ██║   ██╔══██║██╔══██╗██╔══██╗  ╚██╔╝  
   ██║   ██║  ██║██████╔╝██████╔╝   ██║   
   ╚═╝   ╚═╝  ╚═╝╚═════╝ ╚═════╝    ╚═╝   

📄 Version {version}
🚀 Listening at http://{address}
"#
    );
    let listener = tokio::net::TcpListener::bind(address).await.unwrap();

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}

mod chat;
mod completions;
mod events;
mod health;
mod models;
mod server_setting;

pub use chat::*;
pub use completions::*;
pub use events::*;
pub use health::*;
pub use models::*;
pub use server_setting::*;
