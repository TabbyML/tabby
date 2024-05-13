mod routes;
mod services;

mod download;
mod serve;

#[cfg(feature = "ee")]
mod worker;

#[cfg(target_family = "unix")]
use std::os::unix::fs::PermissionsExt;

use clap::{Parser, Subcommand};
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
    #[clap(hide = true, long)]
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
}

#[derive(clap::ValueEnum, strum::Display, PartialEq, Clone)]
pub enum Device {
    #[strum(serialize = "cpu")]
    Cpu,

    #[strum(serialize = "cuda")]
    Cuda,

    #[strum(serialize = "rocm")]
    Rocm,

    #[strum(serialize = "metal")]
    Metal,

    #[strum(serialize = "vulkan")]
    Vulkan,

    #[strum(serialize = "experimental_http")]
    #[clap(hide = true)]
    ExperimentalHttp,
}

impl Device {
    pub fn ggml_use_gpu(&self) -> bool {
        match self {
            Device::Metal | Device::Vulkan | Device::Cuda | Device::Rocm => true,
            Device::Cpu | Device::ExperimentalHttp => false,
        }
    }
}

#[tokio::main]
async fn main() {
    #[cfg(feature = "dep:color-eyre")]
    color_eyre::install().expect("Must be able to install color_eyre");

    let cli = Cli::parse();
    init_logging();

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
        Commands::Scheduler(SchedulerArgs { now, .. }) => {
            tabby_scheduler::scheduler(now, ConfigRepositoryAccess).await
        }
        #[cfg(feature = "ee")]
        Commands::WorkerCompletion(ref args) => {
            worker::main(tabby_webserver::public::WorkerKind::Completion, args).await
        }
        #[cfg(feature = "ee")]
        Commands::WorkerChat(ref args) => {
            worker::main(tabby_webserver::public::WorkerKind::Chat, args).await
        }
    }
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

fn init_logging() {
    let mut layers = Vec::new();

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true)
        .boxed();

    layers.push(fmt_layer);

    let mut dirs = if cfg!(feature = "prod") {
        "tabby=info,otel=debug".into()
    } else {
        "tabby=debug,otel=debug".into()
    };

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
