mod routes;
mod services;

mod download;
mod model;
mod serve;

#[cfg(feature = "ee")]
mod worker;

#[cfg(target_family = "unix")]
use std::os::unix::fs::PermissionsExt;

use clap::{Parser, Subcommand};
use opentelemetry::{
    global,
    sdk::{propagation::TraceContextPropagator, trace, trace::Sampler, Resource},
    KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use tabby_common::config::{Config, ConfigRepositoryAccess};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Open Telemetry endpoint.
    #[clap(long)]
    otlp_endpoint: Option<String>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Starts the api endpoint for IDE / Editor extensions.
    Serve(serve::ServeArgs),

    /// Download the language model for serving.
    Download(download::DownloadArgs),

    /// Run scheduler progress for cron jobs integrating external code repositories.
    Scheduler(SchedulerArgs),

    /// Install, list, and delete models for local use
    #[command(subcommand)]
    Model(model::ModelArgs),

    /// Run completion model as worker
    #[cfg(feature = "ee")]
    #[clap(name = "worker::completion", hide = true)]
    WorkerCompletion(worker::WorkerArgs),

    /// Run chat model as worker
    #[cfg(feature = "ee")]
    #[clap(name = "worker::chat", hide = true)]
    WorkerChat(worker::WorkerArgs),
}

#[derive(clap::Args)]
pub struct SchedulerArgs {
    /// If true, runs scheduler jobs immediately.
    #[clap(long, default_value_t = false)]
    now: bool,

    /// URL to register this worker.
    #[cfg(feature = "ee")]
    #[clap(long)]
    url: Option<String>,

    /// Server token to register this worker to.
    #[cfg(feature = "ee")]
    #[clap(long)]
    token: Option<String>,
}

#[derive(clap::ValueEnum, strum::Display, PartialEq, Clone)]
pub enum Device {
    #[strum(serialize = "cpu")]
    Cpu,

    #[cfg(feature = "cuda")]
    #[strum(serialize = "cuda")]
    Cuda,

    #[cfg(feature = "rocm")]
    #[strum(serialize = "rocm")]
    Rocm,

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[strum(serialize = "metal")]
    Metal,

    #[cfg(feature = "vulkan")]
    #[strum(serialize = "vulkan")]
    Vulkan,

    #[cfg(feature = "experimental-http")]
    #[strum(serialize = "experimental_http")]
    #[clap(hide = true)]
    ExperimentalHttp,
}

impl Device {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn ggml_use_gpu(&self) -> bool {
        *self == Device::Metal
    }

    #[cfg(feature = "cuda")]
    pub fn ggml_use_gpu(&self) -> bool {
        *self == Device::Cuda
    }

    #[cfg(feature = "rocm")]
    pub fn ggml_use_gpu(&self) -> bool {
        *self == Device::Rocm
    }

    #[cfg(feature = "vulkan")]
    pub fn ggml_use_gpu(&self) -> bool {
        *self == Device::Vulkan
    }

    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
    )))]
    pub fn ggml_use_gpu(&self) -> bool {
        false
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    init_logging(cli.otlp_endpoint);

    let config = Config::load().unwrap_or_default();
    let root = tabby_common::path::tabby_root();
    std::fs::create_dir_all(&root).expect("Must be able to create tabby root");
    #[cfg(target_family = "unix")]
    {
        let mut permissions = std::fs::metadata(&root).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&root, permissions).unwrap();
    }

    match cli.command {
        Commands::Serve(ref args) => serve::main(&config, args).await,
        Commands::Download(ref args) => download::main(args).await,
        #[cfg(feature = "ee")]
        Commands::Scheduler(SchedulerArgs {
            now,
            url: Some(url),
            token: Some(token),
        }) => {
            let client = tabby_webserver::public::create_scheduler_client(&url, &token).await;
            tabby_scheduler::scheduler(now, client)
                .await
                .unwrap_or_else(|err| fatal!("Scheduler failed due to '{}'", err))
        }
        Commands::Scheduler(SchedulerArgs { now, .. }) => {
            tabby_scheduler::scheduler(now, ConfigRepositoryAccess)
                .await
                .unwrap_or_else(|err| fatal!("Scheduler failed due to '{}'", err))
        }
        #[cfg(feature = "ee")]
        Commands::WorkerCompletion(ref args) => {
            worker::main(tabby_webserver::public::WorkerKind::Completion, args).await
        }
        #[cfg(feature = "ee")]
        Commands::WorkerChat(ref args) => {
            worker::main(tabby_webserver::public::WorkerKind::Chat, args).await
        }
        Commands::Model(args) => model::main(args).await,
    }

    opentelemetry::global::shutdown_tracer_provider();
}

#[macro_export]
macro_rules! fatal {
    ($msg:expr) => {
        ({
            tracing::error!($msg);
            std::process::exit(1);
        })
    };

    ($fmt:expr, $($arg:tt)*) => {
        ({
            tracing::error!($fmt, $($arg)*);
            std::process::exit(1);
        })
    };
}

fn init_logging(otlp_endpoint: Option<String>) {
    let mut layers = Vec::new();

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true)
        .boxed();

    layers.push(fmt_layer);

    if let Some(otlp_endpoint) = &otlp_endpoint {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp_endpoint),
            )
            .with_trace_config(
                trace::config()
                    .with_resource(Resource::new(vec![KeyValue::new(
                        "service.name",
                        "tabby.server",
                    )]))
                    .with_sampler(Sampler::AlwaysOn),
            )
            .install_batch(opentelemetry::runtime::Tokio);

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
            axum_tracing_opentelemetry::init_propagator().expect("Initializing telemetry");
        };
    }

    let mut dirs = "tabby=info,axum_tracing_opentelemetry=info,otel=debug".to_owned();
    if let Ok(env) = std::env::var(EnvFilter::DEFAULT_ENV) {
        dirs = format!("{dirs},{env}")
    };

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::WARN.into())
        .parse_lossy(dirs);

    tracing_subscriber::registry()
        .with(layers)
        .with(env_filter)
        .init();
}
