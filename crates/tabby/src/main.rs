mod download;
mod serve;

use clap::{Parser, Subcommand};

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
    run: bool,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Serve(args) => serve::main(args).await,
        Commands::Download(args) => download::main(args).await,
        Commands::Scheduler(args) => tabby_scheduler::scheduler(args.run),
    }
}
