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
use tabby_common::{
    config::{Config, SwaggerConfig},
    usage,
};
use tabby_download::Downloader;
use tokio::time::sleep;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use utoipa::{openapi::ServerBuilder, OpenApi};
use utoipa_swagger_ui::SwaggerUi;

use self::{
    engine::{create_engine, EngineInfo},
    health::HealthState,
};
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
        (url = "/", description = "Server"),
    ),
    paths(events::log_event, completions::completions, chat::completions, health::health),
    components(schemas(
        events::LogEventRequest,
        completions::CompletionRequest,
        completions::CompletionResponse,
        completions::Segments,
        completions::Choice,
        chat::ChatCompletionRequest,
        chat::Message,
        chat::ChatCompletionChunk,
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

#[derive(Args)]
pub struct ServeArgs {
    /// Model id for `/completions` API endpoint.
    #[clap(long)]
    model: String,

    /// Model id for `/chat/completions` API endpoints.
    #[clap(long)]
    chat_model: Option<String>,

    #[clap(long, default_value_t = 8080)]
    port: u16,

    /// Device to run model inference.
    #[clap(long, default_value_t=Device::Cpu)]
    device: Device,

    /// GPU indices to run models, only applicable for CUDA.
    #[clap(long, default_values_t=[0])]
    device_indices: Vec<i32>,

    /// DEPRECATED: Do not use.
    #[clap(long)]
    num_replicas_per_device: Option<usize>,

    /// DEPRECATED: Do not use.
    #[clap(long)]
    compute_type: Option<String>,
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
        if let Some(chat_model) = &args.chat_model {
            download_model(chat_model, &args.device).await;
        }
    } else {
        warn!("HTTP device is unstable and does not comply with semver expectations.")
    }

    info!("Starting server, this might takes a few minutes...");

    let mut doc = ApiDoc::openapi();
    doc.override_doc(args, &config.swagger);

    let app = Router::new()
        .merge(api_router(args, config))
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", doc))
        .fallback(fallback());

    let app = if args.chat_model.is_some() {
        app.route("/playground", routing::get(playground::handler))
            .route("/playground/*path", routing::get(playground::handler))
    } else {
        app
    };

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    info!("Listening at {}", address);

    start_heartbeat(args);
    Server::bind(&address)
        .serve(app.into_make_service())
        .await
        .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}

fn api_router(args: &ServeArgs, config: &Config) -> Router {
    let completion_state = {
        let (
            engine,
            EngineInfo {
                prompt_template, ..
            },
        ) = create_engine(&args.model, args);
        let engine = Arc::new(engine);
        let state = completions::CompletionState::new(engine.clone(), prompt_template, config);
        Arc::new(state)
    };

    let chat_state = if let Some(chat_model) = &args.chat_model {
        let (engine, EngineInfo { chat_template, .. }) = create_engine(chat_model, args);
        let Some(chat_template) = chat_template else {
            panic!("Chat model requires specifying prompt template");
        };
        let engine = Arc::new(engine);
        let state = chat::ChatState::new(engine, chat_template);
        Some(Arc::new(state))
    } else {
        None
    };

    let router = Router::new()
        .route("/v1/events", routing::post(events::log_event))
        .route(
            "/v1/health",
            routing::post(health::health).with_state(Arc::new(health::HealthState::new(args))),
        )
        .route(
            "/v1/completions",
            routing::post(completions::completions).with_state(completion_state),
        );

    let router = if let Some(chat_state) = chat_state {
        router.route(
            "/v1beta/chat/completions",
            routing::post(chat::completions).with_state(chat_state),
        )
    } else {
        router
    };

    router
        .layer(CorsLayer::permissive())
        .layer(opentelemetry_tracing_layer())
}

fn fallback() -> routing::MethodRouter {
    routing::get(|| async { axum::response::Redirect::temporary("/swagger-ui") })
}

fn valid_args(args: &ServeArgs) {
    if args.num_replicas_per_device.is_some() {
        warn!("--num-replicas-per-device is deprecated and will be removed in future release.");
    }

    if args.device == Device::Cpu && (args.device_indices.len() != 1 || args.device_indices[0] != 0)
    {
        fatal!("CPU device only supports device indices = [0]");
    }

    if args.compute_type.is_some() {
        warn!("--compute-type is deprecated and will be removed in future release.");
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

trait OpenApiOverride {
    fn override_doc(&mut self, args: &ServeArgs, config: &SwaggerConfig);
}

impl OpenApiOverride for utoipa::openapi::OpenApi {
    fn override_doc(&mut self, args: &ServeArgs, config: &SwaggerConfig) {
        if let Some(servers) = self.servers.as_mut() {
            if let Some(server_url) = &config.server_url {
                servers.push(
                    ServerBuilder::new()
                        .url(server_url)
                        .description(Some("Swagger Server"))
                        .build(),
                );
            }
        }

        if args.chat_model.is_none() {
            self.paths.paths.remove("/v1beta/chat/completions");

            if let Some(components) = self.components.as_mut() {
                components.schemas.remove("ChatCompletionRequest");
                components.schemas.remove("ChatCompletionChunk");
                components.schemas.remove("Message");
            }
        }
    }
}
