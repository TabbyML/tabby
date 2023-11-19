use std::{
    env::consts::ARCH,
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use anyhow::Result;
use axum::{routing, Router};
use axum_prometheus::PrometheusMetricLayer;
use axum_tracing_opentelemetry::opentelemetry_tracing_layer;
use clap::Args;
use hyper::Server;
use tabby_webserver::api::{tracing_context, HubClient, WorkerKind};
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

use crate::{
    fatal, routes,
    services::{
        chat::create_chat_service,
        completion::create_completion_service,
        health::{read_cpu_info, read_cuda_devices},
        model::download_model_if_needed,
    },
    Device,
};

#[derive(Args)]
pub struct WorkerArgs {
    /// URL to register this worker.
    #[clap(long)]
    url: String,

    #[clap(long, default_value_t = 8080)]
    port: u16,

    /// Server token to register this worker to.
    #[clap(long)]
    token: String,

    /// Model id
    #[clap(long, help_heading=Some("Model Options"))]
    model: String,

    /// Device to run model inference.
    #[clap(long, default_value_t=Device::Cpu, help_heading=Some("Model Options"))]
    device: Device,

    /// Parallelism for model serving - increasing this number will have a significant impact on the
    /// memory requirement e.g., GPU vRAM.
    #[clap(long, default_value_t = 1, help_heading=Some("Model Options"))]
    parallelism: u8,
}

async fn make_chat_route(context: WorkerContext, args: &WorkerArgs) -> Router {
    context.register(WorkerKind::Chat, args).await;

    let chat_state =
        Arc::new(create_chat_service(&args.model, &args.device, args.parallelism).await);

    Router::new().route(
        "/v1beta/chat/completions",
        routing::post(routes::chat_completions).with_state(chat_state),
    )
}

async fn make_completion_route(context: WorkerContext, args: &WorkerArgs) -> Router {
    context.register(WorkerKind::Completion, args).await;

    let code = Arc::new(context.client.clone());
    let logger = Arc::new(context.client);
    let completion_state = Arc::new(
        create_completion_service(code, logger, &args.model, &args.device, args.parallelism).await,
    );

    Router::new().route(
        "/v1/completions",
        routing::post(routes::completions).with_state(completion_state),
    )
}

pub async fn main(kind: WorkerKind, args: &WorkerArgs) {
    download_model_if_needed(&args.model).await;

    info!("Starting worker, this might takes a few minutes...");

    let context = WorkerContext::new(&args.url).await;

    let (prometheus_layer, prometheus_handle) = PrometheusMetricLayer::pair();

    let app = match kind {
        WorkerKind::Completion => make_completion_route(context, args).await,
        WorkerKind::Chat => make_chat_route(context, args).await,
    };

    let app = app
        .route(
            "/v1/metrics",
            routing::get(routes::metrics).with_state(Arc::new(prometheus_handle)),
        )
        .layer(CorsLayer::permissive())
        .layer(opentelemetry_tracing_layer())
        .layer(prometheus_layer);

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    info!("Listening at {}", address);

    Server::bind(&address)
        .serve(app.into_make_service())
        .await
        .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}

struct WorkerContext {
    client: HubClient,
}

impl WorkerContext {
    async fn new(url: &str) -> Self {
        Self {
            client: tabby_webserver::api::create_client(url).await,
        }
    }

    async fn register(&self, kind: WorkerKind, args: &WorkerArgs) {
        if let Err(err) = self.register_impl(kind, args).await {
            warn!("Failed to register worker: {}", err)
        }
    }

    async fn register_impl(&self, kind: WorkerKind, args: &WorkerArgs) -> Result<()> {
        let (cpu_info, cpu_count) = read_cpu_info();
        let cuda_devices = read_cuda_devices().unwrap_or_default();
        let worker = self
            .client
            .register_worker(
                tracing_context(),
                kind,
                args.port as i32,
                args.model.to_owned(),
                args.device.to_string(),
                ARCH.to_string(),
                cpu_info,
                cpu_count as i32,
                cuda_devices,
                args.token.clone(),
            )
            .await??;

        info!("Worker alive at {}", worker.addr);

        Ok(())
    }
}
