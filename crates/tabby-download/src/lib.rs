//! Responsible for downloading ML models for use with tabby.
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::Path,
};

use aim_downloader::{bar::WrappedBar, error::DownloadError, https};
use anyhow::{anyhow, bail, Result};
use tabby_common::registry::{parse_model_id, ModelInfo, ModelRegistry};
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
    registry.save_model_info(name);

    let model_path = registry.get_model_path(name);
    if model_path.exists() {
        if !prefer_local_file {
            info!("Checking model integrity..");
            let checksum = sha256::try_digest(&model_path)?;
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

    if !model_info.partition_urls.is_none() {
        return download_split_model(&model_info, &model_path).await;
    }

    let registry = std::env::var("TABBY_DOWNLOAD_HOST").unwrap_or("huggingface.co".to_owned());
    let Some(model_url) = model_info
        .urls
        .iter()
        .flatten()
        .find(|x| x.contains(&registry))
    else {
        return Err(anyhow!(
            "Invalid mirror <{}> for model urls: {:?}",
            registry,
            model_info.urls
        ));
    };

    let strategy = ExponentialBackoff::from_millis(100).map(jitter).take(2);
    let download_job = Retry::spawn(strategy, || download_file(&model_url, model_path.as_path()));
    download_job.await?;
    Ok(())
}

async fn download_split_model(model_info: &ModelInfo, model_path: &Path) -> Result<()> {
    if !model_info.urls.is_none() {
        return Err(anyhow!(
            "{}: Cannot specify both `urls` and `partition_urls`",
            model_info.name
        ));
    }
    let mut paths = vec![];
    let partition_urls = model_info.partition_urls.clone().unwrap_or_default();
    for (index, url) in partition_urls.iter().enumerate() {
        let ext = format!(
            "{}.{}",
            model_path.extension().unwrap_or_default().to_string_lossy(),
            index.to_string()
        );
        let path = model_path.with_extension(ext);
        info!(
            "Downloading {path:?} ({index} / {total})",
            index = index + 1,
            total = partition_urls.len()
        );
        let strategy = ExponentialBackoff::from_millis(100).map(jitter).take(2);
        let download_job = Retry::spawn(strategy, || download_file(url, &path));
        download_job.await?;
        paths.push(path);
    }
    info!("Merging split model files...");
    println!("{model_path:?}");
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(model_path)?;
    for path in paths {
        let mut reader = BufReader::new(File::open(&path)?);
        loop {
            let buffer = reader.fill_buf()?;
            file.write_all(buffer)?;
            let len = buffer.len();
            reader.consume(len);
            if len == 0 {
                break;
            }
        }
        std::fs::remove_file(path)?;
    }
    Ok(())
}

async fn download_file(url: &str, path: &Path) -> Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| anyhow!("Must not be in root directory"))?;
    fs::create_dir_all(dir)?;

    let filename = path
        .to_str()
        .ok_or_else(|| anyhow!("Could not convert filename to UTF-8"))?;
    let intermediate_filename = filename.to_owned() + ".tmp";

    let mut bar = WrappedBar::new(0, url, false);

    if let Err(e) = https::HTTPSHandler::get(url, &intermediate_filename, &mut bar, "").await {
        match e {
            DownloadError::HttpError { name, code } => {
                bail!("Fetching '{name}' failed: Server returned {code} HTTP status")
            }
            DownloadError::Validate { source } => {
                bail!("Failed to validate '{source}'")
            }
        }
    }

    fs::rename(intermediate_filename, filename)?;
    Ok(())
}

pub async fn download_model(model_id: &str, prefer_local_file: bool) {
    let (registry, name) = parse_model_id(model_id);

    let registry = ModelRegistry::new(registry).await;

    let handler = |err| panic!("Failed to fetch model '{}' due to '{}'", model_id, err);
    download_model_impl(&registry, name, prefer_local_file)
        .await
        .unwrap_or_else(handler)
}
