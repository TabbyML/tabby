use std::{net::IpAddr, sync::Arc, time::Duration};

use axum::{routing, Extension, Router};
use clap::Args;
use hyper::StatusCode;
use spinners::{Spinner, Spinners, Stream};
use tabby_common::{
    api::{self, code::CodeSearch, event::EventLogger},
    axum::AllowedCodeRepository,
    config::{Config, ModelConfig},
    usage,
};
use tabby_download::ModelKind;
use tabby_inference::ChatCompletionStream;
use tokio::{sync::oneshot::Sender, time::sleep};
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
        self,
        code::create_code_search,
        completion::{self, create_completion_service_and_chat, CompletionService},
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
    paths(routes::log_event, routes::completions, routes::chat_completions_utoipa, routes::health, routes::setting),
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
        health::HealthState,
        health::Version,
        api::server_setting::ServerSetting,
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
    no_webserver: bool,
}

pub async fn main(config: &Config, args: &ServeArgs) {
    let config = merge_args(config, args);

    load_model(&config).await;

    let tx = try_run_spinner();

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

    #[cfg(feature = "ee")]
    if let Some(ws) = &ws {
        logger = ws.logger();
    }

    let index_reader_provider = Arc::new(IndexReaderProvider::default());
    let docsearch = Arc::new(services::structured_doc::create(
        embedding.clone(),
        index_reader_provider.clone(),
    ));

    let code = Arc::new(create_code_search(
        embedding.clone(),
        index_reader_provider.clone(),
    ));

    let commit_history_search = Arc::new(services::commit::create(
        embedding.clone(),
        index_reader_provider.clone(),
    ));

    let model = &config.model;
    let (completion, completion_stream, chat) = create_completion_service_and_chat(
        &config.completion,
        code.clone(),
        logger.clone(),
        model.completion.clone(),
        model.chat.clone(),
    )
    .await;

    let mut api = api_router(
        args,
        &config,
        logger.clone(),
        code.clone(),
        completion,
        chat.clone(),
        webserver,
    )
    .await;
    let mut ui = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .fallback(|| async { axum::response::Redirect::temporary("/swagger-ui") });

    #[cfg(feature = "ee")]
    if let Some(ws) = &ws {
        let (new_api, new_ui) = ws
            .attach(
                &config,
                api,
                ui,
                code,
                chat,
                completion_stream,
                docsearch,
                |x| Box::new(services::structured_doc::create_serper(x)),
                commit_history_search,
            )
            .await;
        api = new_api;
        ui = new_ui;
    };

    if let Some(tx) = tx {
        tx.send(())
            .unwrap_or_else(|_| warn!("Spinner channel is closed"));
    }
    start_heartbeat(args, &config, webserver);
    run_app(api, Some(ui), args.host, args.port).await
}

async fn load_model(config: &Config) {
    if let Some(ModelConfig::Local(ref model)) = config.model.completion {
        download_model_if_needed(&model.model_id, ModelKind::Completion).await;
    }

    if let Some(ModelConfig::Local(ref model)) = config.model.chat {
        download_model_if_needed(&model.model_id, ModelKind::Chat).await;
    }

    if let ModelConfig::Local(ref model) = config.model.embedding {
        download_model_if_needed(&model.model_id, ModelKind::Embedding).await;
    }
}

async fn api_router(
    args: &ServeArgs,
    config: &Config,
    logger: Arc<dyn EventLogger>,
    _code: Arc<dyn CodeSearch>,
    completion_state: Option<CompletionService>,
    chat_state: Option<Arc<dyn ChatCompletionStream>>,
    webserver: Option<bool>,
) -> Router {
    let mut routers = vec![];

    let health_state = Arc::new(health::HealthState::new(
        &config.model,
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
            .route("/v1beta/models", routing::get(routes::models))
            .with_state(Arc::new(config.clone().into()))
    });

    if let Some(completion_state) = completion_state {
        let mut router = Router::new()
            .route(
                "/v1/completions",
                routing::post(routes::completions).with_state(Arc::new(completion_state)),
            )
            .layer(TimeoutLayer::new(Duration::from_secs(
                config.server.completion_timeout,
            )));

        if webserver.is_none() || webserver.is_some_and(|x| !x) {
            router = router.layer(Extension(AllowedCodeRepository::new_from_config()));
        }

        routers.push(router);
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

fn try_run_spinner() -> Option<Sender<()>> {
    if cfg!(feature = "prod") {
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::task::spawn(async move {
            let mut sp = Spinner::with_timer_and_stream(
                Spinners::Dots,
                "Starting...".into(),
                Stream::Stdout,
            );
            let _ = rx.await;
            sp.stop_with_message("".into());
        });
        Some(tx)
    } else {
        debug!("Starting server, this might take a few minutes...");
        None
    }
}
