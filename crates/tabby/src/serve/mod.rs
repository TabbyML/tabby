mod chat;
mod completions;
mod engine;
mod events;
mod health;
mod search;
mod ui;

use std::{
    fs,
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use axum::{routing, Router, Server};
use axum_tracing_opentelemetry::opentelemetry_tracing_layer;
use clap::Args;
use tabby_common::{
    api::code::{Hit, HitDocument, SearchResponse},
    config::Config,
    usage,
};
use tabby_download::download_model;
use tokio::time::sleep;
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer};
use tracing::{info, warn};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use self::{
    engine::{create_engine, EngineInfo},
    health::HealthState,
};
use crate::{chat::ChatService, fatal, search::create_code_search};
use crate::completions::CompletionService;

#[derive(OpenApi)]
#[openapi(
    info(title="Tabby Server",
        description = "
[![tabby stars](https://img.shields.io/github/stars/TabbyML/tabby)](https://github.com/TabbyML/tabby)
[![Join Slack](https://shields.io/badge/Join-Tabby%20Slack-red?logo=slack)](https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA)

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
    paths(events::log_event, completions::completions, chat::completions, health::health, search::search),
    components(schemas(
        events::LogEventRequest,
        crate::completions::CompletionRequest,
        crate::completions::CompletionResponse,
        crate::completions::Segments,
        crate::completions::Choice,
        crate::completions::Snippet,
        crate::completions::DebugOptions,
        crate::completions::DebugData,
        crate::chat::ChatCompletionRequest,
        crate::chat::Message,
        crate::chat::ChatCompletionChunk,
        health::HealthState,
        health::Version,
        SearchResponse,
        Hit,
        HitDocument
    ))
)]
struct ApiDoc;

#[derive(clap::ValueEnum, strum::Display, PartialEq, Clone)]
pub enum Device {
    #[strum(serialize = "cpu")]
    Cpu,

    #[cfg(feature = "cuda")]
    #[strum(serialize = "cuda")]
    Cuda,

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[strum(serialize = "metal")]
    Metal,

    #[strum(serialize = "experimental_http")]
    ExperimentalHttp,
}

impl Device {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn ggml_use_gpu(&self) -> bool {
        *self == Device::Metal
    }

    #[cfg(feature = "cuda")]
    fn ggml_use_gpu(&self) -> bool {
        *self == Device::Cuda
    }

    #[cfg(not(any(all(target_os = "macos", target_arch = "aarch64"), feature = "cuda")))]
    fn ggml_use_gpu(&self) -> bool {
        false
    }
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

    /// Parallelism for model serving - increasing this number will have a significant impact on the
    /// memory requirement e.g., GPU vRAM.
    #[clap(long, default_value_t = 1)]
    parallelism: u8,
}

pub async fn main(config: &Config, args: &ServeArgs) {
    if args.device != Device::ExperimentalHttp {
        if fs::metadata(&args.model).is_ok() {
            info!("Loading model from local path {}", &args.model);
        } else {
            download_model(&args.model, true).await;
            if let Some(chat_model) = &args.chat_model {
                download_model(chat_model, true).await;
            }
        }
    } else {
        warn!("HTTP device is unstable and does not comply with semver expectations.")
    }

    info!("Starting server, this might takes a few minutes...");

    let mut doc = ApiDoc::openapi();
    doc.override_doc(args);

    let app = Router::new()
        .route("/", routing::get(ui::handler))
        .merge(api_router(args, config).await)
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", doc))
        .fallback(ui::handler);

    let app = if args.chat_model.is_some() {
        app.route("/playground", routing::get(ui::handler))
            .route("/playground.txt", routing::get(ui::handler))
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

async fn api_router(args: &ServeArgs, config: &Config) -> Router {
    let code = Arc::new(create_code_search());
    let completion_state = {
        let (
            engine,
            EngineInfo {
                prompt_template, ..
            },
        ) = create_engine(&args.model, args).await;
        let engine = Arc::new(engine);
        let state =
            CompletionService::new(engine.clone(), code.clone(), prompt_template);
        Arc::new(state)
    };

    let chat_state = if let Some(chat_model) = &args.chat_model {
        let (engine, EngineInfo { chat_template, .. }) = create_engine(chat_model, args).await;
        let Some(chat_template) = chat_template else {
            panic!("Chat model requires specifying prompt template");
        };
        let engine = Arc::new(engine);
        let state = ChatService::new(engine, chat_template);
        Some(Arc::new(state))
    } else {
        None
    };

    let mut routers = vec![];

    let health_state = Arc::new(health::HealthState::new(args));
    routers.push({
        Router::new()
            .route("/v1/events", routing::post(events::log_event))
            .route(
                "/v1/health",
                routing::post(health::health).with_state(health_state.clone()),
            )
            .route(
                "/v1/health",
                routing::get(health::health).with_state(health_state),
            )
    });

    routers.push({
        Router::new()
            .route(
                "/v1/completions",
                routing::post(completions::completions).with_state(completion_state),
            )
            .layer(TimeoutLayer::new(Duration::from_secs(
                config.server.completion_timeout,
            )))
    });

    if let Some(chat_state) = chat_state {
        routers.push({
            Router::new().route(
                "/v1beta/chat/completions",
                routing::post(chat::completions).with_state(chat_state),
            )
        })
    }

    routers.push({
        Router::new().route(
            "/v1beta/search",
            routing::get(search::search).with_state(code),
        )
    });

    let mut root = Router::new();
    for router in routers {
        root = root.merge(router);
    }
    root.layer(CorsLayer::permissive())
        .layer(opentelemetry_tracing_layer())
}

fn start_heartbeat(args: &ServeArgs) {
    let state = HealthState::new(args);
    tokio::spawn(async move {
        loop {
            usage::capture("ServeHealth", &state).await;
            sleep(Duration::from_secs(3000)).await;
        }
    });
}

trait OpenApiOverride {
    fn override_doc(&mut self, args: &ServeArgs);
}

impl OpenApiOverride for utoipa::openapi::OpenApi {
    fn override_doc(&mut self, args: &ServeArgs) {
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
