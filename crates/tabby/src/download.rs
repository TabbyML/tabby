use clap::Args;
use tabby_download::download_model;
use tracing::info;

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
    download_model(&args.model, args.prefer_local_file).await;
    info!("model '{}' is ready", args.model);
}
