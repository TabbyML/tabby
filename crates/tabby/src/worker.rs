use std::{
    env::consts::ARCH,
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use axum::{routing, Router};
use clap::{Args, ArgGroup};
use graphql_client::{reqwest::post_graphql, GraphQLQuery};
use hyper::Server;
use tracing::{info, warn};

use self::register_worker::WorkerKind;
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
pub struct RegisterWorker;

#[derive(Args)]
pub struct WorkerArgs {
    /// URL to register this worker.
    #[clap(long)]
    url: String,

    /// Token to authenticate the worker.
    #[clap(long)]
    token: String,

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

pub async fn chat(args: &WorkerArgs) {
    download_model_if_needed(&args.model).await;

    info!("Starting worker, this might takes a few minutes...");

    let state = Arc::new(create_chat_service(&args.model, &args.device, args.parallelism).await);

    let app = Router::new().route(
        "/v1beta/chat/completions",
        routing::post(routes::chat_completions).with_state(state),
    );

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    info!("Listening at {}", address);

    request_register(WorkerKind::CHAT, args).await;
    Server::bind(&address)
        .serve(app.into_make_service())
        .await
        .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}

pub async fn completion(args: &WorkerArgs) {
    download_model_if_needed(&args.model).await;
    let code = Arc::new(code::create_code_search());
    let logger = Arc::new(event::create_null_logger());
    info!("Starting worker, this might takes a few minutes...");

    let state = Arc::new(
        create_completion_service(code, logger, &args.model, &args.device, args.parallelism).await,
    );

    let app = Router::new().route(
        "/v1/completions",
        routing::post(routes::completions).with_state(state),
    );

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port));
    info!("Listening at {}", address);

    request_register(WorkerKind::COMPLETION, args).await;
    Server::bind(&address)
        .serve(app.into_make_service())
        .await
        .unwrap_or_else(|err| fatal!("Error happens during serving: {}", err))
}

async fn request_register(kind: WorkerKind, args: &WorkerArgs) {
    request_register_impl(
        kind,
        args.url.clone(),
        args.token.clone(),
        args.port as i64,
        args.model.to_owned(),
        args.device.to_string(),
    )
    .await;
}

async fn request_register_impl(
    kind: register_worker::WorkerKind,
    url: String,
    token: String,
    port: i64,
    name: String,
    device: String,
) {
    let client = reqwest::Client::new();
    let (cpu_info, cpu_count) = read_cpu_info();
    let cuda_devices = read_cuda_devices().unwrap_or_default();
    let variables = register_worker::Variables {
        token,
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
