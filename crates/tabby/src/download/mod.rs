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

    /// If true, skip checking for remote model file.
    #[clap(long, default_value_t=true)] 
    prefer_local: bool
}

error_chain! {
    links {
        Another(metadata::Error, metadata::ErrorKind);
    }

    foreign_links {
        Io(std::io::Error);
        HttpRequest(reqwest::Error);
        TemplateError(indicatif::style::TemplateError);
    }
}

pub async fn main(args: &DownloadArgs) -> Result<()> {
    download_model(&args.model, args.prefer_local).await.unwrap();
    Ok(())
}

impl metadata::Metadata {
    async fn download(&mut self, model_id: &str, path: &str, prefer_local: bool) -> Result<()> {
        // Create url.
        let url = format!("https://huggingface.co/{}/resolve/main/{}", model_id, path);

        // Create destination path.
        let filepath = ModelDir::new(model_id).path_string(path);

        // Cache hit.
        let mut cache_hit = false;
        if fs::metadata(&filepath).is_ok() && self.has_etag(&url) {
            if prefer_local || self.match_etag(&url, path).await? {
                cache_hit = true
            }
        }

        if !cache_hit {
            let etag = download_file(&url, &filepath).await?;
            self.update_etag(&url, &etag).await
        }

        Ok(())
    }
}

async fn download_model(model_id: &str, prefer_local: bool) -> Result<()> {
    let mut metadata = metadata::Metadata::from(model_id).await?;

    metadata.download(model_id, "tokenizer.json", prefer_local).await?;
    metadata.download(model_id, "ctranslate2/config.json", prefer_local).await?;
    metadata.download(model_id, "ctranslate2/vocabulary.txt", prefer_local).await?;
    metadata.download(model_id, "ctranslate2/shared_vocabulary.txt", prefer_local).await?;
    metadata.download(model_id, "ctranslate2/model.bin", prefer_local).await?;
    metadata.save(model_id)?;
    Ok(())
}

async fn download_file(url: &str, path: &str) -> Result<String> {
    fs::create_dir_all(Path::new(path).parent().unwrap())?;

    // Reqwest setup
    let res = reqwest::get(url)
        .await
        .or(Err(format!("Failed to GET from '{}'", url)))?;

    let etag = res.headers().get("etag")
        .ok_or(ErrorKind::Msg(format!("Failed to get etag from '{}", url)))?
        .to_str()
        .map_err(|_| -> Error { ErrorKind::Msg(format!("Failed to convert etag to string")).into() } )?
        .to_string();

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
    return Ok(etag);
}
