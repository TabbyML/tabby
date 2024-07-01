use std::{net::IpAddr, sync::Arc, time::Duration};

use axum::{routing, Router};
use clap::Args;
use hyper::StatusCode;
use tabby_common::{
    api::{self, code::CodeSearch, event::EventLogger},
    config::{Config, ConfigAccess, ModelConfig, StaticConfigAccess},
    usage,
};
use tabby_inference::Embedding;
use tokio::time::sleep;
use tower_http::timeout::TimeoutLayer;
use tracing::{debug, warn};
use utoipa::{
    openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme},
    Modify, OpenApi,
};
use utoipa_swagger_ui::SwaggerUi;

use crate::{
    routes::{self, run_app},
    services::{
        self, answer,
        chat::{self, create_chat_service},
        code::create_code_search,
        completion::{self, create_completion_service},
        embedding,
        event::create_event_logger,
        health,
        model::download_model_if_needed,
        tantivy::IndexReaderProvider,
    },
    to_local_config, Device,
};

#[derive(OpenApi)]
#[openapi(
    info(title="Tabby Server",
        description = "
[![tabby stars](https://img.shields.io/github/stars/TabbyML/tabby)](https://github.com/TabbyML/tabby)
[![Join Slack](https://shields.io/badge/Join-Tabby%20Slack-red?logo=slack)](https://links.tabbyml.com/join-slack)

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
    paths(routes::log_event, routes::completions, routes::chat_completions, routes::health, routes::answer, routes::setting),
    components(schemas(
        api::event::LogEventRequest,
        completion::CompletionRequest,
        completion::CompletionResponse,
        completion::Segments,
        completion::Declaration,
        completion::Choice,
        completion::Snippet,
        completion::DebugOptions,
        completion::DebugData,
        chat::ChatCompletionRequest,
        chat::ChatCompletionChoice,
        chat::ChatCompletionDelta,
        api::chat::Message,
        chat::ChatCompletionChunk,
        health::HealthState,
        health::Version,
        api::code::CodeSearchDocument,
        api::code::CodeSearchQuery,
        api::doc::DocSearchDocument,
        answer::AnswerRequest,
        answer::AnswerResponseChunk,
        api::server_setting::ServerSetting
    )),
    modifiers(&SecurityAddon),
)]
struct ApiDoc;

#[derive(Args)]
pub struct ServeArgs {
    /// Model id for `/completions` API endpoint.
    #[clap(long)]
    model: Option<String>,

    /// Model id for `/chat/completions` API endpoints.
    #[clap(long)]
    chat_model: Option<String>,

    #[clap(long, default_value = "0.0.0.0")]
    host: IpAddr,

    #[clap(long, default_value_t = 8080)]
    port: u16,

    /// Device to run model inference.
    #[clap(long, default_value_t=Device::Cpu)]
    device: Device,

    /// Device to run chat model [default equals --device arg]
    #[clap(long, requires("chat_model"))]
    chat_device: Option<Device>,

    /// Parallelism for model serving - increasing this number will have a significant impact on the
    /// memory requirement e.g., GPU vRAM.
    #[clap(long, default_value_t = 1)]
    parallelism: u8,

    #[cfg(feature = "ee")]
    #[clap(hide = true, long, default_value_t = false)]
    #[deprecated(since = "0.11.0", note = "webserver is enabled by default")]
    webserver: bool,

    #[cfg(feature = "ee")]
    #[clap(hide = true, long, default_value_t = false)]
    no_webserver: bool,
}

pub async fn main(config: &Config, args: &ServeArgs) {
    let config = merge_args(config, args);

    load_model(&config).await;

    debug!("Starting server, this might take a few minutes...");

    #[cfg(feature = "ee")]
    #[allow(deprecated)]
    if args.webserver {
        warn!("'--webserver' is enabled by default since 0.11, and will be removed in the next major release. Please remove this flag from your command.");
    }

    #[allow(unused_assignments)]
    let mut webserver = None;

    #[cfg(feature = "ee")]
    {
        webserver = Some(!args.no_webserver)
    }

    let embedding = embedding::create(&config.model.embedding).await;

    #[cfg(feature = "ee")]
    let ws = if !args.no_webserver {
        Some(
            tabby_webserver::public::Webserver::new(create_event_logger(), embedding.clone()).await,
        )
    } else {
        None
    };

    let mut logger: Arc<dyn EventLogger> = Arc::new(create_event_logger());
    let mut config_access: Arc<dyn ConfigAccess> = Arc::new(StaticConfigAccess);

    #[cfg(feature = "ee")]
    if let Some(ws) = &ws {
        logger = ws.logger();
        config_access = ws.clone();
    }

    let index_reader_provider = Arc::new(IndexReaderProvider::default());

    let code = Arc::new(create_code_search(
        config_access,
        embedding.clone(),
        index_reader_provider.clone(),
    ));
    let mut api = api_router(
        args,
        &config,
        logger.clone(),
        code.clone(),
        embedding,
        index_reader_provider,
        webserver,
    )
    .await;
    let mut ui = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .fallback(|| async { axum::response::Redirect::temporary("/swagger-ui") });

    #[cfg(feature = "ee")]
    if let Some(ws) = &ws {
        let (new_api, new_ui) = ws.attach(api, ui, code, config.model.chat.is_some()).await;
        api = new_api;
        ui = new_ui;
    };

    start_heartbeat(args, &config, webserver);
    run_app(api, Some(ui), args.host, args.port).await
}

async fn load_model(config: &Config) {
    if let Some(ModelConfig::Local(ref model)) = config.model.completion {
        download_model_if_needed(&model.model_id).await;
    }

    if let Some(ModelConfig::Local(ref model)) = config.model.chat {
        download_model_if_needed(&model.model_id).await;
    }

    if let ModelConfig::Local(ref model) = config.model.embedding {
        download_model_if_needed(&model.model_id).await;
    }
}

async fn api_router(
    args: &ServeArgs,
    config: &Config,
    logger: Arc<dyn EventLogger>,
    code: Arc<dyn CodeSearch>,
    embedding: Arc<dyn Embedding>,
    index_reader_provider: Arc<IndexReaderProvider>,
    webserver: Option<bool>,
) -> Router {
    let model = &config.model;
    let completion_state = if let Some(completion) = &model.completion {
        Some(Arc::new(
            create_completion_service(code.clone(), logger.clone(), completion).await,
        ))
    } else {
        None
    };

    let chat_state = if let Some(chat) = &model.chat {
        Some(Arc::new(create_chat_service(logger.clone(), chat).await))
    } else {
        None
    };

    let docsearch_state = Arc::new(services::doc::create(embedding, index_reader_provider));

    let answer_state = chat_state.as_ref().map(|chat| {
        Arc::new(services::answer::create(
            chat.clone(),
            code.clone(),
            docsearch_state.clone(),
        ))
    });

    let mut routers = vec![];

    let health_state = Arc::new(health::HealthState::new(
        model,
        &args.device,
        args.chat_model
            .as_deref()
            .map(|_| args.chat_device.as_ref().unwrap_or(&args.device)),
        webserver,
    ));

    routers.push({
        Router::new()
            .route(
                "/v1/events",
                routing::post(routes::log_event).with_state(logger),
            )
            .route(
                "/v1/health",
                routing::post(routes::health).with_state(health_state.clone()),
            )
            .route(
                "/v1/health",
                routing::get(routes::health).with_state(health_state),
            )
    });

    if let Some(completion_state) = completion_state {
        routers.push({
            Router::new()
                .route(
                    "/v1/completions",
                    routing::post(routes::completions).with_state(completion_state),
                )
                .layer(TimeoutLayer::new(Duration::from_secs(
                    config.server.completion_timeout,
                )))
        });
    } else {
        routers.push({
            Router::new().route(
                "/v1/completions",
                routing::post(StatusCode::NOT_IMPLEMENTED),
            )
        })
    }

    if let Some(chat_state) = chat_state {
        routers.push({
            Router::new().route(
                "/v1/chat/completions",
                routing::post(routes::chat_completions).with_state(chat_state.clone()),
            )
        });

        // For forward compatibility of `/v1beta` route.
        routers.push({
            Router::new().route(
                "/v1beta/chat/completions",
                routing::post(routes::chat_completions).with_state(chat_state),
            )
        });
    } else {
        routers.push({
            Router::new().route(
                "/v1/chat/completions",
                routing::post(StatusCode::NOT_IMPLEMENTED),
            )
        });

        routers.push({
            Router::new().route(
                "/v1beta/chat/completions",
                routing::post(StatusCode::NOT_IMPLEMENTED),
            )
        });
    }

    if let Some(answer_state) = answer_state {
        routers.push({
            Router::new().route(
                "/v1beta/answer",
                routing::post(routes::answer).with_state(answer_state),
            )
        });
    } else {
        routers.push({
            Router::new().route("/v1beta/answer", routing::post(StatusCode::NOT_IMPLEMENTED))
        });
    }

    let server_setting_router =
        Router::new().route("/v1beta/server_setting", routing::get(routes::setting));

    #[cfg(feature = "ee")]
    if args.no_webserver {
        routers.push(server_setting_router)
    }

    #[cfg(not(feature = "ee"))]
    routers.push(server_setting_router);

    let mut root = Router::new();
    for router in routers {
        root = root.merge(router);
    }
    root
}

fn start_heartbeat(args: &ServeArgs, config: &Config, webserver: Option<bool>) {
    let state = Arc::new(health::HealthState::new(
        &config.model,
        &args.device,
        args.chat_model
            .as_deref()
            .map(|_| args.chat_device.as_ref().unwrap_or(&args.device)),
        webserver,
    ));
    tokio::spawn(async move {
        loop {
            usage::capture("ServeHealth", &state).await;
            sleep(Duration::from_secs(3000)).await;
        }
    });
}

struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = &mut openapi.components {
            components.add_security_scheme(
                "token",
                SecurityScheme::Http(
                    HttpBuilder::new()
                        .scheme(HttpAuthScheme::Bearer)
                        .bearer_format("token")
                        .build(),
                ),
            )
        }
    }
}

fn merge_args(config: &Config, args: &ServeArgs) -> Config {
    let mut config = (*config).clone();
    if let Some(model) = &args.model {
        if config.model.completion.is_some() {
            warn!("Overriding completion model from config.toml. The overriding behavior might surprise you. Consider setting the model in config.toml directly.");
        }
        config.model.completion = Some(to_local_config(model, args.parallelism, &args.device));
    };

    if let Some(chat_model) = &args.chat_model {
        if config.model.chat.is_some() {
            warn!("Overriding chat model from config.toml. The overriding behavior might surprise you. Consider setting the model in config.toml directly.");
        }
        config.model.chat = Some(to_local_config(
            chat_model,
            args.parallelism,
            args.chat_device.as_ref().unwrap_or(&args.device),
        ));
    }

    config
}
