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

    registry.migrate_q80_model_path(name)?;
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

    let urls = model_info.filter_download_address();

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
        let filename = format!("{:05}-of-{:05}.gguf", index + 1, urls.len());
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

async fn tryget_download_filename(url: &str) -> Result<String> {
    //  try to get filename from Content-Disposition header
    let response = HTTPSHandler::head(url).await?;
    if let Some(content_disposition) = response.get(reqwest::header::CONTENT_DISPOSITION) {
        if let Ok(disposition_str) = content_disposition.to_str() {
            let re = Regex::new(r#"filename="(.+?)""#).unwrap();
            let file_name = re
                .captures(disposition_str)
                .and_then(|cap| cap.get(1))
                .map(|m| m.as_str().to_owned());
            if let Some(file_name) = file_name {
                return Ok(file_name);
            }
        }
    }
    // try to parse filename from URL
    if let Some(parsed_name) = Path::new(url).file_name() {
        let parsed_name = parsed_name.to_string_lossy().to_string();
        if parsed_name.is_empty() {
            Err(anyhow!("Failed to get filename from URL {}", url))
        } else {
            Ok(parsed_name)
        }
    } else {
        Err(anyhow!("Failed to get filename from URL {}", url))
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
