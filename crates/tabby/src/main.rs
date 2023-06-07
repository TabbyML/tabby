mod download;
mod serve;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Serve the model
    Serve(serve::ServeArgs),

    /// Download the model
    Download(download::DownloadArgs),

    /// Starts the scheduler process.
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
    tracing_subscriber::fmt::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("tabby=info".parse().unwrap()))
        .init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Serve(args) => serve::main(args).await,
        Commands::Download(args) => download::main(args).await,
        Commands::Scheduler(args) => tabby_scheduler::scheduler(args.now).await
            .unwrap_or_else(|err| fatal!("Scheduler failed due to '{}'", err))
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
