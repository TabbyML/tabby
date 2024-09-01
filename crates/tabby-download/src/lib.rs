//! Responsible for downloading ML models for use with tabby.
use std::{
    fs::{self},
    path::Path,
};

use aim_downloader::{bar::WrappedBar, error::DownloadError, hash::HashChecker, https};
use anyhow::{anyhow, bail, Result};
use tabby_common::registry::{parse_model_id, ModelRegistry};
use tokio_retry::{
    strategy::{jitter, ExponentialBackoff},
    Retry,
};
use tracing::{info, warn};

fn select_by_download_host(url: &String) -> bool {
    if let Ok(host) = std::env::var("TABBY_DOWNLOAD_HOST") {
        url.contains(&host)
    } else {
        true
    }
}

async fn download_model_impl(
    registry: &ModelRegistry,
    name: &str,
    prefer_local_file: bool,
) -> Result<()> {
    let model_info = registry.get_model_info(name);
    registry.save_model_info(name);

    registry.migrate_model_path(name)?;
    let model_path = registry.get_model_path(name);
    if model_path.exists() {
        if !prefer_local_file {
            info!("Checking model integrity..");
            if HashChecker::check(&model_path.display().to_string(), &model_info.sha256).is_ok() {
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

    let Some(model_url) = model_info
        .urls
        .iter()
        .flatten()
        .find(|x| select_by_download_host(x))
    else {
        return Err(anyhow!("No valid url for model <{}>", model_info.name));
    };

    // Replace the huggingface.co domain with the mirror host if it is set.
    let model_url = if let Ok(host) = std::env::var("TABBY_HUGGINGFACE_HOST_OVERRIDE") {
        model_url.replace("huggingface.co", &host)
    } else {
        model_url.to_owned()
    };

    let strategy = ExponentialBackoff::from_millis(100).map(jitter).take(2);
    let download_job = Retry::spawn(strategy, || {
        download_file(&model_url, model_path.as_path(), &model_info.sha256)
    });
    download_job.await?;
    Ok(())
}

async fn download_file(url: &str, path: &Path, expected_sha256: &str) -> Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| anyhow!("Must not be in root directory"))?;
    fs::create_dir_all(dir)?;

    let filename = path
        .to_str()
        .ok_or_else(|| anyhow!("Could not convert filename to UTF-8"))?;
    let intermediate_filename = filename.to_owned() + ".tmp";

    let mut bar = WrappedBar::new(0, url, false);

    if let Err(e) =
        https::HTTPSHandler::get(url, &intermediate_filename, &mut bar, expected_sha256).await
    {
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
