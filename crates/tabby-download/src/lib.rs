//! Responsible for downloading ML models for use with tabby.
use std::{
    fs::{self},
    path::Path,
};
use reqwest;
use aim_downloader::{bar::WrappedBar, error::DownloadError, hash::HashChecker, https::{self, HTTPSHandler}};
use anyhow::{anyhow, bail, Result};
use tabby_common::registry::{parse_model_id, ModelInfo, ModelRegistry};
use tokio_retry::{
    strategy::{jitter, ExponentialBackoff},
    Retry,
};
use regex::Regex;
use tracing::{info, warn};
use futures::future::join_all;

fn filter_download_urls( model_info: &ModelInfo) -> Vec<String> {
    let download_host = tabby_common::env::get_download_host();
    model_info
        .urls
        .iter()
        .flatten()
        .filter_map(|f| 
            if f.contains(&download_host) {
                if let Some(mirror_host) = tabby_common::env::get_huggingface_mirror_host() {
                    Some(f.replace("huggingface.co", &mirror_host))
                } else {
                Some(f.to_owned())
                }
            } else {
                None
            }
        )
        .collect()

}

async fn download_model_impl(
    registry: &ModelRegistry,
    name: &str,
    prefer_local_file: bool,
) -> Result<()> {
    let model_info = registry.get_model_info(name);
    registry.save_model_info(name);

    registry.migrate_model_path(name)?;

    let model_path = registry.get_model_entry_path(name);
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

            fs::remove_dir_all(registry.get_model_store_dir(name))?;
        } else {
            return Ok(());
        }
    }

    let urls = filter_download_urls(&model_info);

    if urls.len() == 0 {
        bail!("No download URLs available for <{}/{}>", registry.name, model_info.name);
    }
    if urls.len() > 1 {
        // if model_info.entrypoint.is_none(){
        //     bail!("Multiple download URLs available for <{}/{}>, but no entrypoint specified", registry.name, model_info.name);
        // }
        if let Some(urls_sha256) = &model_info.urls_sha256 {
            if urls_sha256.len() != urls.len() {
                bail!("Number of urls_sha256 does not match number of URLs for <{}/{}>", registry.name, model_info.name);
            }
        } else {
            bail!("No urls_sha256 available for <{}/{}>", registry.name, model_info.name);
        }
    }
    
    // prepare for download
    let dir = model_path.parent().ok_or_else(|| anyhow!("Must not be in root directory"))?;
    fs::create_dir_all(dir)?;

    let mut urls_sha256 = vec![];
    if urls.len() > 1 {
        urls_sha256.extend(model_info.urls_sha256.clone().unwrap());
    } else {
        urls_sha256.push(model_info.sha256.clone());
    }

    let mut download_tasks = vec![];
    for (url, sha256) in urls.iter().zip(urls_sha256.iter()) {
        let dir = registry.get_model_store_dir(name).to_string_lossy().into_owned();
        let filename = if urls.len() == 1 {
            Some(model_info.entrypoint.clone())
        } else {
            None
        };
        let strategy = ExponentialBackoff::from_millis(100).map(jitter).take(2);
        download_tasks.push(Retry::spawn(strategy,  move || {
                    let dir = dir.clone();
                    let filename = filename.clone();
                    download_file(&url, dir, filename, &sha256)
                }));
    }

    let results = join_all(download_tasks).await;
    
    // handle errors
    let errors: Vec<anyhow::Error> = results.into_iter().filter_map(|e| {
        if let Err(e) = e {
            Some(e)
        } else {
            None
        }
    }).collect();
    if errors.is_empty() {
        Ok(())
    } else {
        let combined_error = errors.into_iter().fold(anyhow::anyhow!("Multiple errors occurred"), |acc, err| acc.context(err));
        Err(combined_error)
    }

}

async fn tryget_download_filename(url: &str) -> Result<String> {
    //  try to get filename from Content-Disposition header
    let response = HTTPSHandler::head(url).await?;
    if let Some(content_disposition) = response.get(reqwest::header::CONTENT_DISPOSITION) {
        if let Ok(disposition_str) = content_disposition.to_str() {
            let re = Regex::new(r#"filename="(.+?)""#).unwrap();
            let file_name = re.captures(disposition_str).
            and_then(|cap| cap.get(1)).
            map(|m| m.as_str().
            to_owned());
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

async fn download_file(url: &str, dir: String, filename:Option<String>, expected_sha256: &str) -> Result<()> {

    let filename = filename.unwrap_or(tryget_download_filename(url).await?);
    let fullpath = format!{"{}{}{}", dir,std::path::MAIN_SEPARATOR ,filename};
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
            urls: Some(vec!["https://huggingface.co/test".to_string(), "https://huggingface.co/test2".to_string(), "https://modelscope.co/test2".to_string()]),
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
            urls: Some(vec!["https://huggingface.co/test".to_string(), "https://modelscope.co/test2".to_string()]),
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