mod otel;
mod routes;
mod services;

mod download;
mod serve;

#[cfg(target_family = "unix")]
use std::os::unix::fs::PermissionsExt;

use clap::{Parser, Subcommand};
use tabby_common::config::{Config, ModelConfig};

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
}

#[tokio::main]
async fn main() {
    color_eyre::install().expect("Must be able to install color_eyre");

    let cli = Cli::parse();
    let _guard = otel::init_tracing_subscriber(cli.otlp_endpoint);

    let config = Config::load().expect("Must be able to load config");
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

fn to_local_config(model: &str, parallelism: u8, device: &Device) -> ModelConfig {
    let num_gpu_layers = if *device != Device::Cpu {
        std::env::var("LLAMA_CPP_N_GPU_LAYERS")
            .map(|s| s.parse::<u16>().ok())
            .ok()
            .flatten()
            .unwrap_or(9999)
    } else {
        0
    };
    // This only works when the the model is included in cli arguments.
    let enable_fast_attention = Some(std::env::var("LLAMA_CPP_FAST_ATTENTION").is_ok());

    ModelConfig::new_local(model, parallelism, num_gpu_layers, enable_fast_attention)
}
