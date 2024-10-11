//! Responsible for downloading ML models for use with tabby.
use std::{
    fs::{self},
    path::Path,
};

use aim_downloader::{
    bar::WrappedBar,
    error::DownloadError,
    hash::HashChecker,
    https::{self, HTTPSHandler},
};
use anyhow::{anyhow, bail, Result};
use futures::future::join_all;
use regex::Regex;
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
        .partitioned_urls
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

    let mut download_tasks = vec![];
    for (index, url) in urls.iter().enumerate() {
        let dir = registry
            .get_model_store_dir(name)
            .to_string_lossy()
            .into_owned();
        let filename = format!("model-{:05}-of-{:05}.gguf", index + 1, urls.len());
        let strategy = ExponentialBackoff::from_millis(100).map(jitter).take(2);
        download_tasks.push(Retry::spawn(strategy, move || {
            let dir = dir.clone();
            let filename = filename.clone();
            download_file(&url.0, dir, filename, &url.1)
        }));
    }

    let results = join_all(download_tasks).await;

    // handle errors
    let errors: Vec<anyhow::Error> = results
        .into_iter()
        .filter_map(|e| if let Err(e) = e { Some(e) } else { None })
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        let combined_error = errors
            .into_iter()
            .fold(anyhow::anyhow!("Multiple errors occurred"), |acc, err| {
                acc.context(err)
            });
        Err(combined_error)
    }
}

async fn download_file(
    url: &str,
    dir: String,
    filename: String,
    expected_sha256: &str,
) -> Result<()> {
    let fullpath = format! {"{}{}{}", dir, std::path::MAIN_SEPARATOR, filename};
    let intermediate_filename = fullpath.clone() + ".tmp";
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

    use super::*;
    #[test]
    fn test_filter_download_urls() {
        // multiple urls
        let model_info = ModelInfo {
            name: "test".to_string(),
            urls: Some(vec![
                "https://huggingface.co/test".to_string(),
                "https://huggingface.co/test2".to_string(),
                "https://modelscope.co/test2".to_string(),
            ]),
            urls_sha256: Some(vec!["test_sha256".to_string(), "test2_sha256".to_string()]),
            entrypoint: "test".to_string(),
            sha256: "test_sha256".to_string(),
            prompt_template: None,
            chat_template: None,
        };
        let urls = super::filter_download_urls(&model_info);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://huggingface.co/test");
        assert_eq!(urls[1], "https://huggingface.co/test2");

        // single url
        let model_info = ModelInfo {
            name: "test".to_string(),
            urls: Some(vec![
                "https://huggingface.co/test".to_string(),
                "https://modelscope.co/test2".to_string(),
            ]),
            urls_sha256: None,
            entrypoint: "model.gguf".to_string(),
            sha256: "test_sha256".to_string(),
            prompt_template: None,
            chat_template: None,
        };
        let urls = super::filter_download_urls(&model_info);
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0], "https://huggingface.co/test");
    }

    #[tokio::test]
    async fn test_tryget_download_filename() {
        let url = "https://huggingface.co/TabbyML/models/resolve/main/.gitattributes";
        let result = tryget_download_filename(url).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ".gitattributes");
    }
}
