mod completions;
mod context;
mod events;
mod health;

use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use axum::{routing, Router, Server};
use axum_tracing_opentelemetry::opentelemetry_tracing_layer;
use clap::Args;
use tabby_common::{config::Config, usage};
use tokio::time::sleep;
use tower_http::cors::CorsLayer;
use tracing::info;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use self::{context::TabbyContext, health::HealthState};
use crate::fatal;

#[derive(OpenApi)]
#[openapi(
    info(title="Tabby Server",
        description = "
[![tabby stars](https://img.shields.io/github/stars/TabbyML/tabby?style=social)](https://github.com/TabbyML/tabby)

OpenAPI documentation for [tabby](https://github.com/TabbyML/tabby), a self-hosted AI coding assistant.",
        license(name = "Apache 2.0", url="https://github.com/TabbyML/tabby/blob/main/LICENSE")
    ),
    servers(
        (url = "https://playground.app.tabbyml.com", description = "Playground server"),
        (url = "http://localhost:8080", description = "Local server"),
    ),
    paths(events::log_event, completions::completion, health::health),
    components(schemas(
        events::LogEventRequest,
        completions::CompletionRequest,
        completions::CompletionResponse,
        completions::Segments,
        completions::Choice,
        health::HealthState,
    ))
)]
struct ApiDoc;

#[derive(clap::ValueEnum, strum::Display, PartialEq, Clone)]
pub enum Device {
    #[strum(serialize = "cpu")]
    Cpu,

    #[strum(serialize = "cuda")]
    Cuda,
}

#[derive(clap::ValueEnum, strum::Display, PartialEq, Clone)]
#[clap(rename_all = "snake_case")]
pub enum ComputeType {
    /// Use the fastest computation type that is supported on this system and device
    #[strum(serialize = "auto")]
    Auto,

    /// Quantize model weights to use int8 for inference.
    ///
    /// On CUDA devices, embedding / linear layers runs on int8, while other layers runs on
    /// float32.
    #[strum(serialize = "int8")]
    Int8,

    /// Use float16 for inference, only supported on CUDA devices.
    #[strum(serialize = "float16")]
    Float16,

    /// Use int8 / float16 mixed precision for inference, only supported on CUDA devices.
    ///
    /// This mode is the same as int8 for CUDA devices, but all non quantized layers are run in float16
    /// instead of float32.
    #[strum(serialize = "int8_float16")]
    Int8Float16,

    /// Use float32 for inference.
    #[strum(serialize = "float32")]
    Float32,
}

#[derive(Args)]
pub struct ServeArgs {
    /// Model id for serving.
    #[clap(long)]
    model: String,

    #[clap(long, default_value_t = 8080)]
    port: u16,

    /// Device to run model inference.
    #[clap(long, default_value_t=Device::Cpu)]
    device: Device,

    /// GPU indices to run models, only applicable for CUDA.
    #[clap(long, default_values_t=[0])]
    device_indices: Vec<i32>,

    /// Number of replicas per device, only applicable for CPU.
    #[clap(long, default_value_t = 1)]
    num_replicas_per_device: usize,

    /// Compute type
    #[clap(long, default_value_t=ComputeType::Auto)]
    compute_type: ComputeType,
}

pub async fn main(config: &Config, args: &ServeArgs) {
    valid_args(args);
    let mut context = TabbyContext::new();

    // Ensure model exists.
    tabby_download::download_model(&args.model, true)
        .await
        .unwrap_or_else(|err| {
            fatal!(
                "Failed to fetch model due to '{}', is '{}' a valid model id?",
                err,
                args.model
            )
        });

    let app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .nest("/v1", api_router(args, config, &mut context))
        .fallback(fallback());

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    info!("Listening at {}", address);

    start_heartbeat(args, &mut context);
    Server::bind(&address)
        .serve(app.into_make_service())
        .await
        .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}

fn api_router(args: &ServeArgs, config: &Config, context: &mut TabbyContext) -> Router {
    Router::new()
        .route("/events", routing::post(events::log_event))
        .route(
            "/health",
            routing::post(health::health)
                .with_state(Arc::new(health::HealthState::new(args, context))),
        )
        .route(
            "/completions",
            routing::post(completions::completion)
                .with_state(Arc::new(completions::CompletionState::new(args, config))),
        )
        .layer(CorsLayer::permissive())
        .layer(opentelemetry_tracing_layer())
}

fn fallback() -> routing::MethodRouter {
    routing::get(|| async { axum::response::Redirect::temporary("/swagger-ui") })
}

fn valid_args(args: &ServeArgs) {
    if args.device == Device::Cuda && args.num_replicas_per_device != 1 {
        fatal!("CUDA device only supports 1 replicas per device");
    }

    if args.device == Device::Cpu && (args.device_indices.len() != 1 || args.device_indices[0] != 0)
    {
        fatal!("CPU device only supports device indices = [0]");
    }

    if args.device == Device::Cpu && args.compute_type != ComputeType::Int8 {
        match args.compute_type {
            ComputeType::Auto | ComputeType::Int8 => {}
            _ => fatal!("CPU device only supports int8 compute type"),
        }
    }
}

fn start_heartbeat(args: &ServeArgs, context: &mut TabbyContext) {
    let state = HealthState::new(args, context);
    tokio::spawn(async move {
        loop {
            usage::capture("ServeHealth", &state).await;
            sleep(Duration::from_secs(300)).await;
        }
    });
}
