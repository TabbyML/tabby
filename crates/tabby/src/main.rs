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
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Serve(args) => {
            serve::main(args)
                .await
                .expect("Error happens during the serve");
        }
        Commands::Download(args) => {
            download::main(args)
                .await
                .expect("Error happens during the download");
        }
    }
}
