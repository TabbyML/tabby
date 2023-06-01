mod admin;
mod completions;
mod events;

use crate::Cli;
use anyhow::Result;
use axum::{routing, Router, Server};
use clap::{error::ErrorKind, Args, CommandFactory};
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

#[derive(Args)]
pub struct ServeArgs {
    /// Model id for serving.
    #[clap(long)]
    model: String,

    #[clap(long, default_value_t = 8080)]
    port: u16,

    /// Device to run model inference.
    #[clap(long, default_value_t=Device::CPU)]
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

pub async fn main(args: &ServeArgs) -> Result<()> {
    valid_args(args)?;

    // Ensure model exists.
    crate::download::download_model(&args.model, true).await?;

    let app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .nest("/v1", api_router(args))
        .fallback(fallback(args.experimental_admin_panel))
        .layer(CorsLayer::permissive());

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    println!("Listening at {}", address);
    Server::bind(&address)
        .serve(app.into_make_service())
        .await?;
    Ok(())
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

fn valid_args(args: &ServeArgs) -> Result<()> {
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

    Ok(())
}
