use clap::Args;
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
    tabby_download::download_model(
        &args.model,
        /* download_ctranslate2_files= */ true,
        /* download_ggml_files= */ true,
        args.prefer_local_file,
    )
    .await
    .unwrap_or_else(|err| {
        fatal!(
            "Failed to fetch model due to '{}', is '{}' a valid model id?",
            err,
            args.model
        )
    });
    info!("model '{}' is ready", args.model);
}
