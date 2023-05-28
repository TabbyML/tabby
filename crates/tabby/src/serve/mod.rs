mod admin;
mod completions;
mod events;

use crate::Cli;
use axum::{routing, Router, Server};
use clap::{error::ErrorKind, Args, CommandFactory};
use hyper::Error;
use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};
use tower_http::cors::CorsLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

#[derive(OpenApi)]
#[openapi(
    paths(events::log_event, completions::completion,),
    components(schemas(
        events::LogEventRequest,
        completions::CompletionRequest,
        completions::CompletionResponse,
        completions::Choice
    ))
)]
struct ApiDoc;

#[derive(clap::ValueEnum, strum::Display, PartialEq, Clone)]
pub enum Device {
    #[strum(serialize = "cpu")]
    CPU,

    #[strum(serialize = "cuda")]
    CUDA,
}

#[derive(clap::ValueEnum, strum::Display, Clone)]
pub enum ModelType {
    #[strum(serialize = "encoder-decoder")]
    EncoderDecoder,

    #[strum(serialize = "decoder")]
    Decoder,
}

#[derive(Args)]
pub struct ServeArgs {
    /// path to model for serving
    #[clap(long)]
    model: String,

    /// model type for serving
    #[clap(long, default_value_t=ModelType::Decoder)]
    model_type: ModelType,

    #[clap(long, default_value_t = 8080)]
    port: u16,

    #[clap(long, default_value_t=Device::CPU)]
    device: Device,

    #[clap(long, default_values_t=[0])]
    device_indices: Vec<i32>,

    /// num_replicas_per_device
    #[clap(long, default_value_t = 1)]
    num_replicas_per_device: usize,

    #[clap(long, default_value_t = false)]
    experimental_admin_panel: bool,
}

pub async fn main(args: &ServeArgs) -> Result<(), Error> {
    valid_args(args);
    let app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .nest("/v1", api_router(args))
        .fallback(fallback(args.experimental_admin_panel));

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    println!("Listening at {}", address);
    Server::bind(&address).serve(app.into_make_service()).await
}

fn api_router(args: &ServeArgs) -> Router {
    Router::new()
        .route("/events", routing::post(events::log_event))
        .route(
            "/completions",
            routing::post(completions::completion)
                .with_state(Arc::new(completions::CompletionState::new(args))),
        )
        .layer(CorsLayer::permissive())
}

fn fallback(experimental_admin_panel: bool) -> routing::MethodRouter {
    if experimental_admin_panel {
        routing::get(admin::handler)
    } else {
        routing::get(|| async { axum::response::Redirect::temporary("/swagger-ui") })
    }
}

fn valid_args(args: &ServeArgs) {
    if args.device == Device::CUDA && args.num_replicas_per_device != 1 {
        Cli::command()
            .error(
                ErrorKind::ValueValidation,
                "CUDA device only supports 1 replicas per device",
            )
            .exit();
    }

    if args.device == Device::CPU && (args.device_indices.len() != 1 || args.device_indices[0] != 0)
    {
        Cli::command()
            .error(
                ErrorKind::ValueValidation,
                "CPU device only supports device indices = [0]",
            )
            .exit();
    }
}
