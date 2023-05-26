use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use axum::{response::Redirect, routing, Router, Server};
use clap::Args;
use ctranslate2_bindings::TextInferenceEngineCreateOptionsBuilder;
use hyper::Error;
use std::path::Path;
use tower_http::cors::CorsLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

mod completions;
mod events;

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

#[derive(clap::ValueEnum, Clone)]
pub enum Device {
    CPU,
    CUDA,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let printable = match *self {
            Device::CPU => "cpu",
            Device::CUDA => "cuda",
        };
        write!(f, "{}", printable)
    }
}

#[derive(Args)]
pub struct ServeArgs {
    /// path to model for serving
    #[clap(long)]
    model: String,

    #[clap(long, default_value_t=8080)]
    port: u16,

    #[clap(long, default_value_t=Device::CPU)]
    device: Device,

    #[clap(long, default_values_t=[0])]
    device_indices: Vec<i32>,

    /// num_replicas_per_device
    #[clap(long, default_value_t = 1)]
    num_replicas_per_device: usize,
}

pub async fn main(args: &ServeArgs) -> Result<(), Error> {
    let device = format!("{}", args.device);
    let options = TextInferenceEngineCreateOptionsBuilder::default()
        .model_path(
            Path::new(&args.model)
                .join(device.clone())
                .display()
                .to_string(),
        )
        .tokenizer_path(
            Path::new(&args.model)
                .join("tokenizer.json")
                .display()
                .to_string(),
        )
        .device(device)
        .device_indices(args.device_indices.clone())
        .num_replicas_per_device(args.num_replicas_per_device)
        .build()
        .unwrap();
    let completions_state = Arc::new(completions::CompletionState::new(options));

    let app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/v1/events", routing::post(events::log_event))
        .route("/v1/completions", routing::post(completions::completion))
        .with_state(completions_state)
        .route(
            "/",
            routing::get(|| async { Redirect::temporary("/swagger-ui") }),
        )
        .layer(CorsLayer::permissive());

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    println!("Listening at {}", address);
    Server::bind(&address).serve(app.into_make_service()).await
}
