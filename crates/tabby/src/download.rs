use clap::Args;
use tabby_download::Downloader;
use tracing::{info, log::warn};

use crate::fatal;

#[derive(Args)]
pub struct DownloadArgs {
    /// model id to fetch.
    #[clap(long)]
    model: String,

    /// If true, skip checking for remote model file.
    #[clap(long, default_value_t = false)]
    prefer_local_file: bool,
}

pub async fn main(args: &DownloadArgs) {
    let downloader = Downloader::new(&args.model, args.prefer_local_file);

    downloader
        .download_ggml_files()
        .await
        .unwrap_or_else(|err| fatal!("Failed to fetch model '{}' due to '{}'", args.model, err));

    info!("model '{}' is ready", args.model);
}
