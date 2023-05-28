use std::cmp;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Args;
use error_chain::error_chain;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Args)]
pub struct DownloadArgs {
    /// model name to fetch.
    #[clap(long)]
    model: String,

    #[clap(long, default_value = "cpu")]
    device: String,
}

error_chain! {
     foreign_links {
         Io(std::io::Error);
         HttpRequest(reqwest::Error);
         TemplateError(indicatif::style::TemplateError);
     }
}

pub async fn main(args: &DownloadArgs) -> Result<()> {
    download_model(&args.model, &args.device).await.unwrap();
    Ok(())
}

async fn download_model(model_id: &str, device: &str) -> Result<()> {
    download_model_file(model_id, "tokenizer.json").await?;
    download_model_file(model_id, &format!("ctranslate2/{}/config.json", device)).await?;
    download_model_file(model_id, &format!("ctranslate2/{}/model.bin", device)).await?;
    download_model_file(
        model_id,
        &format!("ctranslate2/{}/shared_vocabulary.txt", device),
    )
    .await?;
    Ok(())
}

fn get_model_dir(model_id: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    let tabby_root = format!("{}/.tabby", home);
    let model_dir = Path::new(&tabby_root).join("models").join(model_id);
    model_dir
}

async fn download_model_file(model_id: &str, fname: &str) -> Result<()> {
    // Create url.
    let url = format!("https://huggingface.co/{}/resolve/main/{}", model_id, fname);

    // Create destination path.
    let filepath = get_model_dir(model_id).join(fname);

    // Ensure dir.
    fs::create_dir_all(filepath.parent().unwrap())?;
    let path = filepath.display().to_string();

    download_file(&format!("{}/{}", model_id, fname), &url, &path).await
}

async fn download_file(name: &str, url: &str, path: &str) -> Result<()> {
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
    pb.set_message(format!("Downloading {}", &name));

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

    pb.finish_with_message(format!("Downloaded {}", &name));
    return Ok(());
}
