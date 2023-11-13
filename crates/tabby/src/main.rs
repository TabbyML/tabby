mod api;
mod download;
mod routes;
mod serve;
mod services;

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
}

#[derive(clap::Args)]
pub struct SchedulerArgs {
    /// If true, runs scheduler jobs immediately.
    #[clap(long, default_value_t = false)]
    now: bool,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    init_logging(cli.otlp_endpoint);

    let config = Config::load().unwrap_or_default();

    match &cli.command {
        Commands::Serve(args) => serve::main(&config, args).await,
        Commands::Download(args) => download::main(args).await,
        Commands::Scheduler(args) => tabby_scheduler::scheduler(args.now)
            .await
            .unwrap_or_else(|err| fatal!("Scheduler failed due to '{}'", err)),
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
