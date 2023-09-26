use clap::Args;
use tabby_download::Downloader;
use tracing::info;

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

    let handler = |err| fatal!("Failed to fetch model '{}' due to '{}'", args.model, err,);

    downloader
        .download_ctranslate2_files()
        .await
        .unwrap_or_else(handler);
    downloader
        .download_ggml_files()
        .await
        .unwrap_or_else(handler);

    info!("model '{}' is ready", args.model);
}
