use clap::Args;
use tracing::info;

use crate::services::model::Downloader;

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
    let mut downloader = Downloader::new();
    let (registry, _) = downloader
        .get_model_registry_and_info(&args.model)
        .await
        .unwrap();
    downloader
        .download_model(&registry, &args.model, args.prefer_local_file)
        .await
        .unwrap();
    info!("model '{}' is ready", args.model);
}
