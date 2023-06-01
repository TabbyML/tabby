mod metadata;

use std::cmp;
use std::fs;
use std::io::Write;
use std::path::Path;

use clap::Args;
use error_chain::error_chain;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use tabby_common::path::ModelDir;

#[derive(Args)]
pub struct DownloadArgs {
    /// model id to fetch.
    #[clap(long)]
    model: String,
}

error_chain! {
     foreign_links {
         Io(std::io::Error);
         HttpRequest(reqwest::Error);
         TemplateError(indicatif::style::TemplateError);
     }
}

pub async fn main(args: &DownloadArgs) -> Result<()> {
    download_model(&args.model).await.unwrap();
    Ok(())
}

async fn download_model(model_id: &str) -> Result<()> {
    download_metadata(model_id).await?;
    download_model_file(model_id, "tokenizer.json").await?;
    download_model_file(model_id, &format!("ctranslate2/config.json")).await?;
    download_model_file(model_id, &format!("ctranslate2/vocabulary.txt")).await?;
    download_model_file(model_id, &format!("ctranslate2/shared_vocabulary.txt")).await?;
    download_model_file(model_id, &format!("ctranslate2/model.bin")).await?;
    Ok(())
}

async fn download_metadata(model_id: &str) -> Result<()> {
    let url = format!("https://huggingface.co/api/models/{}", model_id);
    let filepath = ModelDir::new(model_id).metadata_file();
    download_file(&url, &filepath).await
}

async fn download_model_file(model_id: &str, fname: &str) -> Result<()> {
    // Create url.
    let url = format!("https://huggingface.co/{}/resolve/main/{}", model_id, fname);

    // Create destination path.
    let filepath = ModelDir::new(model_id).path_string(fname);
    download_file(&url, &filepath).await
}

async fn download_file(url: &str, path: &str) -> Result<()> {
    fs::create_dir_all(Path::new(path).parent().unwrap())?;

    // Reqwest setup
    let res = reqwest::get(url)
        .await
        .or(Err(format!("Failed to GET from '{}'", url)))?;

    let total_size = res
        .content_length()
        .ok_or(format!("Failed to get content length from '{}'", url))?;

    // Indicatif setup
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
        .progress_chars("#>-"));
    pb.set_message(format!("Downloading {}", path));

    // download chunks
    let mut file = fs::File::create(&path).or(Err(format!("Failed to create file '{}'", &path)))?;
    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item.or(Err(format!("Error while downloading file")))?;
        file.write_all(&chunk)
            .or(Err(format!("Error while writing to file")))?;
        let new = cmp::min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }

    pb.finish_with_message(format!("Downloaded {}", path));
    return Ok(());
}
