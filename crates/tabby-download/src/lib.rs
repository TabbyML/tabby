use std::{cmp, fs, io::Write, path::Path};

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use tabby_common::registry::{parse_model_id, ModelRegistry};
use tokio_retry::{
    strategy::{jitter, ExponentialBackoff},
    Retry,
};
use tracing::{info, warn};

async fn download_model_impl(
    registry: &ModelRegistry,
    name: &str,
    prefer_local_file: bool,
) -> Result<()> {
    let model_info = registry.get_model_info(name);
    let model_path = registry.get_model_path(name);
    if model_path.exists() {
        if !prefer_local_file {
            info!("Checking model integrity..");
            let checksum = sha256::try_digest(&model_path).unwrap();
            if checksum == model_info.sha256 {
                return Ok(());
            }

            warn!(
                "Checksum doesn't match for <{}/{}>, re-downloading...",
                registry.name, name
            );
            fs::remove_file(&model_path)?;
        } else {
            return Ok(());
        }
    }

    let registry = std::env::var("TABBY_DOWNLOAD_HOST").unwrap_or("huggingface.co".to_owned());
    let Some(model_url) = model_info.urls.iter().find(|x| x.contains(&registry)) else {
        return Err(anyhow!(
            "Invalid mirror <{}> for model urls: {:?}",
            registry,
            model_info.urls
        ));
    };

    let strategy = ExponentialBackoff::from_millis(100).map(jitter).take(2);
    let download_job = Retry::spawn(strategy, || download_file(model_url, model_path.as_path()));
    download_job.await?;
    Ok(())
}

async fn download_file(url: &str, path: &Path) -> Result<()> {
    fs::create_dir_all(path.parent().unwrap())?;

    // Reqwest setup
    let res = reqwest::get(url).await?;

    if !res.status().is_success() {
        return Err(anyhow!(format!("Invalid url: {}", url)));
    }

    let total_size = res
        .content_length()
        .ok_or(anyhow!("No content length in headers"))?;

    // Indicatif setup
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
        .progress_chars("#>-"));
    pb.set_message(format!("Downloading {}", path.display()));

    // download chunks
    let mut file = fs::File::create(path)?;
    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item?;
        file.write_all(&chunk)?;
        let new = cmp::min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }

    pb.finish_with_message(format!("Downloaded {}", path.display()));
    Ok(())
}

pub async fn download_model(model_id: &str, prefer_local_file: bool) {
    // Local file path.
    if fs::metadata(model_id).is_ok() {
        return;
    }

    let (registry, name) = parse_model_id(model_id);

    let registry = ModelRegistry::new(registry).await;

    let handler = |err| panic!("Failed to fetch model '{}' due to '{}'", model_id, err);
    download_model_impl(&registry, name, prefer_local_file)
        .await
        .unwrap_or_else(handler)
}
