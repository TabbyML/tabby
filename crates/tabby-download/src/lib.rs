//! Responsible for downloading ML models for use with tabby.
use std::fs;

use aim_downloader::{
    bar::{MultiBar, WrappedBar},
    error::DownloadError,
    hash::HashChecker,
    https,
};
use anyhow::{bail, Result};
use tabby_common::registry::{parse_model_id, ModelInfo, ModelRegistry};
use tokio_retry::{
    strategy::{jitter, ExponentialBackoff},
    Retry,
};
use tracing::{info, warn};

pub fn get_download_host() -> String {
    std::env::var("TABBY_DOWNLOAD_HOST").unwrap_or_else(|_| "huggingface.co".to_string())
}

pub fn get_huggingface_mirror_host() -> Option<String> {
    std::env::var("TABBY_HUGGINGFACE_HOST_OVERRIDE").ok()
}

pub fn filter_download_address(model_info: &ModelInfo) -> Vec<(String, String)> {
    let download_host = get_download_host();
    if let Some(urls) = &model_info.urls {
        if !urls.is_empty() {
            let url = model_info
                .urls
                .iter()
                .flatten()
                .find(|f| f.contains(&download_host));
            if let Some(url) = url {
                if let Some(mirror_host) = get_huggingface_mirror_host() {
                    return vec![(
                        url.replace("huggingface.co", &mirror_host),
                        model_info.sha256.clone(),
                    )];
                }
                return vec![(url.to_owned(), model_info.sha256.clone())];
            }
        }
    };

    model_info
        .addresses
        .iter()
        .flatten()
        .map(|x| -> (String, String) {
            let url = x.urls.iter().find(|f| f.contains(&download_host));
            if let Some(url) = url {
                if let Some(mirror_host) = get_huggingface_mirror_host() {
                    return (
                        url.replace("huggingface.co", &mirror_host),
                        x.sha256.clone(),
                    );
                }
                return (url.to_owned(), x.sha256.clone());
            }
            panic!("No download URLs available for <{}>", model_info.name);
        })
        .collect()
}

async fn download_model_impl(
    registry: &ModelRegistry,
    name: &str,
    prefer_local_file: bool,
) -> Result<()> {
    let model_info = registry.get_model_info(name);
    registry.save_model_info(name);

    registry.migrate_relative_model_path(name)?;

    let model_path = registry.get_model_entry_path(name);
    if let Some(model_path) = model_path {
        if !prefer_local_file {
            info!("Checking model integrity..");
            if HashChecker::check(&model_path.display().to_string(), &model_info.sha256).is_ok() {
                return Ok(());
            }
            warn!(
                "Checksum doesn't match for <{}/{}>, re-downloading...",
                registry.name, name
            );

            fs::remove_dir_all(registry.get_model_store_dir(name))?;
        } else {
            return Ok(());
        }
    }

    let urls = filter_download_address(model_info);

    if urls.is_empty() {
        bail!(
            "No download URLs available for <{}/{}>",
            registry.name,
            model_info.name
        );
    }

    // prepare for download
    let dir = registry.get_model_store_dir(name);
    fs::create_dir_all(dir)?;

    // let progressbars = MultiBar::new(false);
    let progressbars = MultiBar::new(false);

    for (index, url) in urls.iter().enumerate() {
        let dir = registry
            .get_model_store_dir(name)
            .to_string_lossy()
            .into_owned();
        let filename = format!("model-{:05}-of-{:05}.gguf", index + 1, urls.len());
        let strategy = ExponentialBackoff::from_millis(100).map(jitter).take(2);
        let progressbars = progressbars.clone();

        Retry::spawn(strategy, move || {
            let dir = dir.clone();
            let filename = filename.clone();
            let bar = progressbars.create_bar(0, &url.0).unwrap();

            download_file(&url.0, dir, filename, &url.1, bar)
        })
        .await?;
    }

    Ok(())
}

async fn download_file(
    url: &str,
    dir: String,
    filename: String,
    expected_sha256: &str,
    bar: WrappedBar,
) -> Result<()> {
    let fullpath = format! {"{}{}{}", dir, std::path::MAIN_SEPARATOR, filename};
    let intermediate_filename = fullpath.clone() + ".tmp";
    let mut bar = bar.clone();
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

    fs::rename(intermediate_filename, fullpath)?;
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

#[cfg(test)]
mod tests {
    use tabby_common::registry::ModelInfo;

    
    #[test]
    fn test_filter_download_address() {
        // multiple urls
        let model_info = ModelInfo {
            name: "test".to_string(),
            urls: Some(vec![
                "https://huggingface.co/test".to_string(),
                "https://huggingface.co/test2".to_string(),
                "https://modelscope.co/test2".to_string(),
            ]),
            sha256: "test_sha256".to_string(),
            prompt_template: None,
            chat_template: None,
            addresses: None,
        };
        let urls = super::filter_download_address(&model_info);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0].0, "https://huggingface.co/test");
        assert_eq!(urls[1].0, "https://huggingface.co/test2");

        // single url
        let model_info = ModelInfo {
            name: "test".to_string(),
            urls: Some(vec![
                "https://huggingface.co/test".to_string(),
                "https://modelscope.co/test2".to_string(),
            ]),
            sha256: "test_sha256".to_string(),
            prompt_template: None,
            chat_template: None,
            addresses: None,
        };
        let urls = super::filter_download_address(&model_info);
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].0, "https://huggingface.co/test");
    }
}
