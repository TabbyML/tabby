mod admin;
mod completions;
mod events;

use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use axum::{routing, Router, Server};
use clap::Args;
use tower_http::cors::CorsLayer;
use tracing::info;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::fatal;

#[derive(OpenApi)]
#[openapi(
    info(title="Tabby Server", description = "OpenAPI documentation for [tabby](https://github.com/TabbyML/tabby), a self-hosted AI coding assistant. ![tabby stars](https://img.shields.io/github/stars/TabbyML/tabby?style=social)", license(name = "Apache 2.0", url="https://github.com/TabbyML/tabby/blob/main/LICENSE")),
    servers(
        (url = "https://app.tabbyml.com/api/workspace/tabbyml/tabby", description = "Local server"),
        (url = "http://localhost:8080", description = "Local server"),
    ),
    paths(events::log_event, completions::completion,),
    components(schemas(
        events::LogEventRequest,
        completions::CompletionRequest,
        completions::CompletionResponse,
        completions::Segments,
        completions::Choice
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

    /// *INTERNAL ONLY*
    #[clap(long, default_value_t = false)]
    experimental_admin_panel: bool,
}

pub async fn main(args: &ServeArgs) {
    valid_args(args);

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
        .nest("/v1", api_router(args))
        .fallback(fallback(args.experimental_admin_panel))
        .layer(CorsLayer::permissive());

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    info!("Listening at {}", address);
    Server::bind(&address)
        .serve(app.into_make_service())
        .await
        .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}

fn api_router(args: &ServeArgs) -> Router {
    Router::new()
        .route("/events", routing::post(events::log_event))
        .route(
            "/completions",
            routing::post(completions::completion)
                .with_state(Arc::new(completions::CompletionState::new(args))),
        )
}

fn fallback(experimental_admin_panel: bool) -> routing::MethodRouter {
    if experimental_admin_panel {
        routing::get(admin::handler)
    } else {
        routing::get(|| async { axum::response::Redirect::temporary("/swagger-ui") })
    }
}

fn valid_args(args: &ServeArgs) {
    if args.device == Device::Cuda && args.num_replicas_per_device != 1 {
        fatal!("CUDA device only supports 1 replicas per device");
    }

    if args.device == Device::Cpu && (args.device_indices.len() != 1 || args.device_indices[0] != 0)
    {
        fatal!("CPU device only supports device indices = [0]");
    }
}
