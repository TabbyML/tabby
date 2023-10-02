mod chat;
mod completions;
mod engine;
mod events;
mod health;
mod playground;

use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use axum::{routing, Router, Server};
use axum_tracing_opentelemetry::opentelemetry_tracing_layer;
use clap::Args;
use tabby_common::{config::Config, usage};
use tabby_download::Downloader;
use tokio::time::sleep;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use utoipa::{openapi::ServerBuilder, OpenApi};
use utoipa_swagger_ui::SwaggerUi;

use self::{engine::create_engine, health::HealthState};
use crate::fatal;

#[derive(OpenApi)]
#[openapi(
    info(title="Tabby Server",
        description = "
[![tabby stars](https://img.shields.io/github/stars/TabbyML/tabby)](https://github.com/TabbyML/tabby)
[![Join Slack](https://shields.io/badge/Tabby-Join%20Slack-red?logo=slack)](https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA)

Install following IDE / Editor extensions to get started with [Tabby](https://github.com/TabbyML/tabby).
* [VSCode Extension](https://github.com/TabbyML/tabby/tree/main/clients/vscode) – Install from the [marketplace](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby), or [open-vsx.org](https://open-vsx.org/extension/TabbyML/vscode-tabby)
* [VIM Extension](https://github.com/TabbyML/tabby/tree/main/clients/vim)
* [IntelliJ Platform Plugin](https://github.com/TabbyML/tabby/tree/main/clients/intellij) – Install from the [marketplace](https://plugins.jetbrains.com/plugin/22379-tabby)
",
        license(name = "Apache 2.0", url="https://github.com/TabbyML/tabby/blob/main/LICENSE")
    ),
    servers(
        (url = "https://playground.app.tabbyml.com", description = "Playground server"),
    ),
    paths(events::log_event, completions::completions, chat::completions, health::health),
    components(schemas(
        events::LogEventRequest,
        completions::CompletionRequest,
        completions::CompletionResponse,
        completions::Segments,
        completions::Choice,
        chat::ChatCompletionRequest,
        chat::ChatCompletionResponse,
        health::HealthState,
        health::Version,
    ))
)]
struct ApiDoc;

#[derive(clap::ValueEnum, strum::Display, PartialEq, Clone)]
pub enum Device {
    #[strum(serialize = "cpu")]
    Cpu,

    #[strum(serialize = "cuda")]
    Cuda,

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[strum(serialize = "metal")]
    Metal,

    #[strum(serialize = "experimental_http")]
    ExperimentalHttp,
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
    /// Model id for `/completions` API endpoint.
    #[clap(long)]
    model: String,

    /// Model id for `/chat/completions` API endpoints.
    #[clap(long)]
    instruct_model: Option<String>,

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

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn should_download_ggml_files(_device: &Device) -> bool {
    false
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn should_download_ggml_files(device: &Device) -> bool {
    *device == Device::Metal
}

pub async fn main(config: &Config, args: &ServeArgs) {
    valid_args(args);

    if args.device != Device::ExperimentalHttp {
        download_model(&args.model, &args.device).await;
        if let Some(instruct_model) = &args.instruct_model {
            download_model(instruct_model, &args.device).await;
        }
    } else {
        warn!("HTTP device is unstable and does not comply with semver expectations.")
    }

    info!("Starting server, this might takes a few minutes...");

    let doc = add_localhost_server(ApiDoc::openapi(), args.port);
    let doc = add_proxy_server(doc, config.swagger.server_url.clone());
    let app = api_router(args, config)
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", doc))
        .route("/playground", routing::get(playground::handler))
        .route("/playground/*path", routing::get(playground::handler))
        .fallback(fallback());

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    info!("Listening at {}", address);

    start_heartbeat(args);
    Server::bind(&address)
        .serve(app.into_make_service())
        .await
        .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}

fn api_router(args: &ServeArgs, config: &Config) -> Router {
    let (engine, prompt_template) = create_engine(&args.model, args);
    let engine = Arc::new(engine);
    let instruct_engine = if let Some(instruct_model) = &args.instruct_model {
        Arc::new(create_engine(instruct_model, args).0)
    } else {
        engine.clone()
    };

    Router::new()
        .route("/v1/events", routing::post(events::log_event))
        .route(
            "/v1/health",
            routing::post(health::health).with_state(Arc::new(health::HealthState::new(args))),
        )
        .route(
            "/v1/completions",
            routing::post(completions::completions).with_state(Arc::new(
                completions::CompletionState::new(engine.clone(), prompt_template, config),
            )),
        )
        .route(
            "/v1beta/chat/completions",
            routing::post(chat::completions)
                .with_state(Arc::new(chat::ChatState::new(instruct_engine.clone()))),
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

fn start_heartbeat(args: &ServeArgs) {
    let state = HealthState::new(args);
    tokio::spawn(async move {
        loop {
            usage::capture("ServeHealth", &state).await;
            sleep(Duration::from_secs(300)).await;
        }
    });
}

fn add_localhost_server(doc: utoipa::openapi::OpenApi, port: u16) -> utoipa::openapi::OpenApi {
    let mut doc = doc;
    if let Some(servers) = doc.servers.as_mut() {
        servers.push(
            ServerBuilder::new()
                .url(format!("http://localhost:{}", port))
                .description(Some("Local server"))
                .build(),
        );
    }

    doc
}

fn add_proxy_server(
    doc: utoipa::openapi::OpenApi,
    server_url: Option<String>,
) -> utoipa::openapi::OpenApi {
    if server_url.is_none() {
        return doc;
    }

    let server_url: String = server_url.unwrap();
    let mut doc = doc;
    if let Some(servers) = doc.servers.as_mut() {
        servers.push(
            ServerBuilder::new()
                .url(server_url)
                .description(Some("Swagger Server"))
                .build(),
        );
    }

    doc
}

async fn download_model(model: &str, device: &Device) {
    let downloader = Downloader::new(model, /* prefer_local_file= */ true);
    let handler = |err| fatal!("Failed to fetch model '{}' due to '{}'", model, err,);
    let download_result = if should_download_ggml_files(device) {
        downloader.download_ggml_files().await
    } else {
        downloader.download_ctranslate2_files().await
    };

    download_result.unwrap_or_else(handler);
}
