mod routes;
mod services;

mod download;
mod serve;

#[cfg(feature = "ee")]
mod worker;

use clap::{Parser, Subcommand};
use opentelemetry::{
    global,
    sdk::{propagation::TraceContextPropagator, trace, trace::Sampler, Resource},
    KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use tabby_common::config::Config;
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

    /// Run completion model as worker
    #[cfg(feature = "ee")]
    #[clap(name = "worker::completion")]
    WorkerCompletion(worker::WorkerArgs),

    /// Run chat model as worker
    #[cfg(feature = "ee")]
    #[clap(name = "worker::chat")]
    WorkerChat(worker::WorkerArgs),

    /// Execute the repository sync job.
    #[cfg(feature = "ee")]
    #[clap(name = "job::sync")]
    JobSync,

    /// Execute the index job.
    #[cfg(feature = "ee")]
    #[clap(name = "job::index")]
    JobIndex,
}

#[derive(clap::Args)]
pub struct SchedulerArgs {
    /// If true, runs scheduler jobs immediately.
    #[clap(long, default_value_t = false)]
    now: bool,
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

    #[cfg(feature = "experimental-http")]
    #[strum(serialize = "experimental_http")]
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

    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        feature = "cuda",
        feature = "rocm",
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

    match &cli.command {
        Commands::Serve(args) => serve::main(&config, args).await,
        Commands::Download(args) => download::main(args).await,
        Commands::Scheduler(args) => tabby_scheduler::scheduler(args.now, &config)
            .await
            .unwrap_or_else(|err| fatal!("Scheduler failed due to '{}'", err)),
        #[cfg(feature = "ee")]
        Commands::JobSync => tabby_scheduler::job_sync(&config),
        #[cfg(feature = "ee")]
        Commands::JobIndex => tabby_scheduler::job_index(&config),
        #[cfg(feature = "ee")]
        Commands::WorkerCompletion(args) => {
            worker::main(tabby_webserver::public::WorkerKind::Completion, args).await
        }
        #[cfg(feature = "ee")]
        Commands::WorkerChat(args) => {
            worker::main(tabby_webserver::public::WorkerKind::Chat, args).await
        }
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
            axum_tracing_opentelemetry::init_propagator().unwrap();
        };
    }

    let env_filter = EnvFilter::from_default_env()
        .add_directive("tabby=info".parse().unwrap())
        .add_directive("axum_tracing_opentelemetry=info".parse().unwrap())
        .add_directive("otel=debug".parse().unwrap());

    tracing_subscriber::registry()
        .with(layers)
        .with(env_filter)
        .init();
}
