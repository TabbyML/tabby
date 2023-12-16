use std::{net::IpAddr, sync::Arc, time::Duration};

use axum::{routing, Router};
use clap::Args;
use hyper::StatusCode;
use tabby_common::{
    api,
    api::{code::CodeSearch, event::EventLogger},
    config::Config,
    usage,
};
use tokio::time::sleep;
use tower_http::timeout::TimeoutLayer;
use tracing::info;
use utoipa::{
    openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme},
    Modify, OpenApi,
};
use utoipa_swagger_ui::SwaggerUi;

use crate::{
    routes::{self, run_app},
    services::{
        chat::{self, create_chat_service},
        code::create_code_search,
        completion::{self, create_completion_service},
        event::create_logger,
        health,
        model::download_model_if_needed,
    },
    Device,
};

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
    paths(routes::log_event, routes::completions, routes::chat_completions, routes::health, routes::search),
    components(schemas(
        api::event::LogEventRequest,
        completion::CompletionRequest,
        completion::CompletionResponse,
        completion::Segments,
        completion::Choice,
        completion::Snippet,
        completion::DebugOptions,
        completion::DebugData,
        chat::ChatCompletionRequest,
        chat::Message,
        chat::ChatCompletionChunk,
        health::HealthState,
        health::Version,
        api::code::SearchResponse,
        api::code::Hit,
        api::code::HitDocument
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

    /// Parallelism for model serving - increasing this number will have a significant impact on the
    /// memory requirement e.g., GPU vRAM.
    #[clap(long, default_value_t = 1)]
    parallelism: u8,

    #[cfg(feature = "ee")]
    #[clap(hide = true, long, default_value_t = false)]
    webserver: bool,
}

pub async fn main(config: &Config, args: &ServeArgs) {
    #[cfg(feature = "experimental-http")]
    if args.device == Device::ExperimentalHttp {
        tracing::warn!("HTTP device is unstable and does not comply with semver expectations.");
    } else {
        load_model(args).await;
    }
    #[cfg(not(feature = "experimental-http"))]
    load_model(args).await;

    info!("Starting server, this might take a few minutes...");

    let logger = Arc::new(create_logger());
    let code = Arc::new(create_code_search());

    let api = api_router(args, config, logger.clone(), code.clone()).await;
    let ui = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()));

    #[cfg(feature = "ee")]
    let (api, ui) = if args.webserver {
        tabby_webserver::attach_webserver(api, ui, logger, code).await
    } else {
        let ui = ui.fallback(|| async { axum::response::Redirect::temporary("/swagger-ui") });
        (api, ui)
    };

    #[cfg(not(feature = "ee"))]
    let ui = ui.fallback(|| async { axum::response::Redirect::temporary("/swagger-ui") });

    start_heartbeat(args);
    run_app(api, Some(ui), args.host, args.port).await
}

async fn load_model(args: &ServeArgs) {
    if let Some(model) = &args.model {
        download_model_if_needed(model).await;
    }

    if let Some(chat_model) = &args.chat_model {
        download_model_if_needed(chat_model).await
    }
}

async fn api_router(
    args: &ServeArgs,
    config: &Config,
    logger: Arc<dyn EventLogger>,
    code: Arc<dyn CodeSearch>,
) -> Router {
    let completion_state = if let Some(model) = &args.model {
        Some(Arc::new(
            create_completion_service(
                code.clone(),
                logger.clone(),
                model,
                &args.device,
                args.parallelism,
            )
            .await,
        ))
    } else {
        None
    };

    let chat_state = if let Some(chat_model) = &args.chat_model {
        Some(Arc::new(
            create_chat_service(chat_model, &args.device, args.parallelism).await,
        ))
    } else {
        None
    };

    let mut routers = vec![];

    let health_state = Arc::new(health::HealthState::new(
        args.model.as_deref(),
        args.chat_model.as_deref(),
        &args.device,
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
                "/v1beta/chat/completions",
                routing::post(routes::chat_completions).with_state(chat_state),
            )
        })
    } else {
        routers.push({
            Router::new().route(
                "/v1beta/chat/completions",
                routing::post(StatusCode::NOT_IMPLEMENTED),
            )
        })
    }

    routers.push({
        Router::new().route(
            "/v1beta/search",
            routing::get(routes::search).with_state(code),
        )
    });

    let mut root = Router::new();
    for router in routers {
        root = root.merge(router);
    }
    root
}

fn start_heartbeat(args: &ServeArgs) {
    let state = health::HealthState::new(
        args.model.as_deref(),
        args.chat_model.as_deref(),
        &args.device,
    );
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
