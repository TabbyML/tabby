use std::{env::consts::ARCH, net::IpAddr, sync::Arc};

use axum::{routing, Router};
use clap::Args;
use tabby_common::{
    api::{code::CodeSearch, event::EventLogger},
    config::Config,
};
use tabby_webserver::public::{RegisterWorkerRequest, WorkerClient, WorkerKind};
use tracing::info;

use crate::{
    routes::{self, run_app},
    services::{
        chat::create_chat_service,
        completion::create_completion_service,
        health::{read_cpu_info, read_cuda_devices},
        model::download_model_if_needed,
    },
    to_local_config, Device,
};

#[derive(Args)]
pub struct WorkerArgs {
    /// URL to register this worker.
    #[clap(long)]
    url: String,

    /// Server token to register this worker to.
    #[clap(long)]
    token: String,

    #[clap(long, default_value = "0.0.0.0")]
    host: IpAddr,

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

async fn make_chat_route(logger: Arc<dyn EventLogger>, config: &Config) -> Router {
    let chat_state = Arc::new(
        create_chat_service(
            logger,
            config
                .model
                .chat
                .as_ref()
                .expect("Chat model config is missing"),
        )
        .await,
    );

    Router::new()
        .route(
            "/v1/chat/completions",
            routing::post(routes::chat_completions).with_state(chat_state.clone()),
        )
        .route(
            "/v1beta/chat/completions",
            routing::post(routes::chat_completions).with_state(chat_state),
        )
}

async fn make_completion_route(
    code: Arc<dyn CodeSearch>,
    logger: Arc<dyn EventLogger>,
    config: &Config,
) -> Router {
    let completion_state = Arc::new(
        create_completion_service(
            code,
            logger,
            config
                .model
                .completion
                .as_ref()
                .expect("Completion model config is missing"),
        )
        .await,
    );

    Router::new().route(
        "/v1/completions",
        routing::post(routes::completions).with_state(completion_state),
    )
}

pub async fn main(config: &Config, kind: WorkerKind, args: &WorkerArgs) {
    download_model_if_needed(&args.model).await;
    let config = merge_args(config, args, &kind);

    info!("Starting worker, this might take a few minutes...");

    let context = WorkerContext::new(kind.clone(), args).await;
    let code = Arc::new(context.client);
    let logger = code.clone();

    let app = match kind {
        WorkerKind::Completion => make_completion_route(code, logger, &config).await,
        WorkerKind::Chat => make_chat_route(logger.clone(), &config).await,
    };

    run_app(app, None, args.host, args.port).await
}

struct WorkerContext {
    client: WorkerClient,
}

impl WorkerContext {
    async fn new(kind: WorkerKind, args: &WorkerArgs) -> Self {
        let (cpu_info, cpu_count) = read_cpu_info();
        let cuda_devices = read_cuda_devices().unwrap_or_default();

        Self {
            client: tabby_webserver::public::create_worker_client(
                &args.url,
                &args.token,
                RegisterWorkerRequest {
                    kind,
                    port: args.port,
                    name: args.model.to_owned(),
                    device: args.device.to_string(),
                    arch: ARCH.to_string(),
                    cpu_info,
                    cpu_count: cpu_count as i32,
                    cuda_devices,
                },
            )
            .await,
        }
    }
}

fn merge_args(config: &Config, args: &WorkerArgs, kind: &WorkerKind) -> Config {
    let mut config = (*config).clone();
    let override_config = Some(to_local_config(&args.model, args.parallelism, &args.device));

    match kind {
        WorkerKind::Chat => config.model.chat = override_config,
        WorkerKind::Completion => config.model.completion = override_config,
    }

    config
}
