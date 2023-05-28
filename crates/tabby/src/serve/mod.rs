use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use axum::{routing, Router, Server};
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

#[derive(clap::ValueEnum, strum::Display, Clone)]
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
    let app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .nest("/v1", api_router(args))
        .fallback(fallback(args));

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    println!("Listening at {}", address);
    Server::bind(&address).serve(app.into_make_service()).await
}

fn api_router(args: &ServeArgs) -> Router {
    Router::new()
        .route("/events", routing::post(events::log_event))
        .route("/completions", routing::post(completions::completion))
        .with_state(Arc::new(new_completion_state(args)))
        .layer(CorsLayer::permissive())
}

mod admin;
fn fallback(args: &ServeArgs) -> routing::MethodRouter {
    if args.experimental_admin_panel {
        routing::get(admin::handler)
    } else {
        routing::get(|| async { axum::response::Redirect::temporary("/swagger-ui") })
    }
}

fn new_completion_state(args: &ServeArgs) -> completions::CompletionState {
    let device = format!("{}", args.device);
    let options = TextInferenceEngineCreateOptionsBuilder::default()
        .model_path(
            Path::new(&args.model)
                .join("ctranslate2")
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
        .model_type(format!("{}", args.model_type))
        .device_indices(args.device_indices.clone())
        .num_replicas_per_device(args.num_replicas_per_device)
        .build()
        .unwrap();
    completions::CompletionState::new(options)
}
