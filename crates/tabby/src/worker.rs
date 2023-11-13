use std::{
    env::consts::ARCH,
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use axum::{routing, Router};
use clap::Args;
use graphql_client::{reqwest::post_graphql, GraphQLQuery};
use hyper::Server;
use tracing::{info, warn};

use crate::{
    fatal, routes,
    services::{
        chat::create_chat_service,
        code,
        completion::create_completion_service,
        event::{self},
        health::{read_cpu_info, read_cuda_devices},
        model::download_model_if_needed,
    },
    Device,
};

#[derive(GraphQLQuery)]
#[graphql(
    schema_path = "../../ee/tabby-webserver/graphql/schema.graphql",
    query_path = "./graphql/worker.query.graphql"
)]
struct RegisterWorker;

#[derive(Args)]
pub struct WorkerArgs {
    /// URL to register this worker.
    #[clap(long)]
    url: String,

    #[clap(long, default_value_t = 8080)]
    port: u16,

    /// Model id
    #[clap(long, help_heading=Some("Model Options"))]
    model: String,

    /// Device to run model inference.
    #[clap(long, default_value_t=Device::Cpu, help_heading=Some("Model Options"))]
    device: Device,

    /// Parallelism for model serving - increasing this number will have a significant impact on the
    /// memory requirement e.g., GPU vRAM.
    #[clap(long, default_value_t = 1, help_heading=Some("Model Options"))]
    parallelism: u8,
}

async fn make_chat_route(args: &WorkerArgs) -> Router {
    let state = Arc::new(create_chat_service(&args.model, &args.device, args.parallelism).await);

    request_register(register_worker::WorkerKind::CHAT, args).await;

    Router::new().route(
        "/v1beta/chat/completions",
        routing::post(routes::chat_completions).with_state(state),
    )
}

async fn make_completion_route(args: &WorkerArgs) -> Router {
    let code = Arc::new(code::create_code_search());
    let logger = Arc::new(event::create_null_logger());
    let state = Arc::new(
        create_completion_service(code, logger, &args.model, &args.device, args.parallelism).await,
    );

    request_register(register_worker::WorkerKind::COMPLETION, args).await;

    Router::new().route(
        "/v1/completions",
        routing::post(routes::completions).with_state(state),
    )
}

pub enum WorkerKind {
    Chat,
    Completion,
}

pub async fn main(kind: WorkerKind, args: &WorkerArgs) {
    download_model_if_needed(&args.model).await;

    info!("Starting worker, this might takes a few minutes...");

    let app = match kind {
        WorkerKind::Completion => make_completion_route(args).await,
        WorkerKind::Chat => make_chat_route(args).await,
    };

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    info!("Listening at {}", address);

    Server::bind(&address)
        .serve(app.into_make_service())
        .await
        .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}

async fn request_register(kind: register_worker::WorkerKind, args: &WorkerArgs) {
    request_register_impl(
        kind,
        args.url.clone(),
        args.port as i64,
        args.model.to_owned(),
        args.device.to_string(),
    )
    .await;
}

async fn request_register_impl(
    kind: register_worker::WorkerKind,
    url: String,
    port: i64,
    name: String,
    device: String,
) {
    let client = reqwest::Client::new();
    let (cpu_info, cpu_count) = read_cpu_info();
    let cuda_devices = read_cuda_devices().unwrap_or_default();
    let variables = register_worker::Variables {
        port,
        kind,
        name,
        device,
        arch: ARCH.to_string(),
        cpu_info,
        cpu_count: cpu_count as i64,
        cuda_devices,
    };

    let url = format!("{}/graphql", url);
    match post_graphql::<RegisterWorker, _>(&client, &url, variables).await {
        Ok(x) => {
            let addr = x.data.unwrap().worker.addr;
            info!("Worker alive at {}", addr);
        }
        Err(err) => warn!("Failed to register worker: {}", err),
    }
}
