use std::{fs, path::Path};

use aim_downloader::{bar::WrappedBar, https};
use anyhow::{anyhow, Result};
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
    registry.save_model_info(name);

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
    let dir = path.parent().unwrap();
    fs::create_dir_all(dir)?;

    let filename = path.to_str().unwrap();
    let intermediate_filename = filename.to_owned() + ".tmp";

    let mut bar = WrappedBar::new(0, url, false);

    https::HTTPSHandler::get(url, &intermediate_filename, &mut bar, "").await?;

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
