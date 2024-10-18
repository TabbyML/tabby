//! Responsible for downloading ML models for use with tabby.
use std::fs;

use aim_downloader::{bar::WrappedBar, error::DownloadError, hash::HashChecker, https};
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
                        model_info.sha256.clone().unwrap_or_default(),
                    )];
                }
                return vec![(
                    url.to_owned(),
                    model_info.sha256.clone().unwrap_or_default(),
                )];
            }
        }
    };

    model_info
        .partition_urls
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

macro_rules! partitioned_file_name {
    ($index:expr, $total:expr) => {
        format!("model-{:05}-of-{:05}.gguf", $index + 1, $total)
    };
}

async fn download_model_impl(
    registry: &ModelRegistry,
    name: &str,
    prefer_local_file: bool,
) -> Result<()> {
    let model_info = registry.get_model_info(name);
    registry.migrate_legacy_model_path(name)?;

    let urls = filter_download_address(model_info);
    if urls.is_empty() {
        bail!(
            "No download URLs available for <{}/{}>",
            registry.name,
            model_info.name
        );
    }

    if !prefer_local_file {
        info!("Checking model integrity..");

        let mut sha256_matched = true;
        for (index, url) in urls.iter().enumerate() {
            if HashChecker::check(
                partitioned_file_name!(index + 1, urls.len()).as_str(),
                &url.1,
            ).is_err()
            {
                sha256_matched = false;
                break;
            }
        }

        if sha256_matched {
            return Ok(());
        }

        warn!(
            "Checksum doesn't match for <{}/{}>, re-downloading...",
            registry.name, name
        );

        fs::remove_dir_all(registry.get_model_dir(name))?;
    }

    // prepare for download
    let dir = registry.get_model_store_dir(name);
    fs::create_dir_all(dir)?;
    registry.save_model_info(name);

    for (index, url) in urls.iter().enumerate() {
        let dir = registry
            .get_model_store_dir(name)
            .to_string_lossy()
            .into_owned();
        let filename: String = partitioned_file_name!(index + 1, urls.len());
        let strategy = ExponentialBackoff::from_millis(100).map(jitter).take(2);

        Retry::spawn(strategy, move || {
            let dir = dir.clone();
            let filename = filename.clone();

            download_file(&url.0, dir, filename, &url.1)
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
    use tabby_common::registry::{ModelInfo, PartitionModelUrl};

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
            sha256: Some("test_sha256".to_string()),
            prompt_template: None,
            chat_template: None,
            partition_urls: None,
        };
        let urls = super::filter_download_address(&model_info);
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].0, "https://huggingface.co/test");

        // single url
        let model_info = ModelInfo {
            name: "test".to_string(),
            urls: Some(vec![
                "https://huggingface.co/test".to_string(),
                "https://modelscope.co/test2".to_string(),
            ]),
            sha256: Some("test_sha256".to_string()),
            prompt_template: None,
            chat_template: None,
            partition_urls: None,
        };
        let urls = super::filter_download_address(&model_info);
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].0, "https://huggingface.co/test");
    }

    #[test]
    fn test_filter_download_address_multiple_partitions() {
        let model_info = ModelInfo {
            name: "test".to_string(),
            urls: None,
            sha256: Some("test_sha256".to_string()),
            prompt_template: None,
            chat_template: None,
            partition_urls: Some(vec![
                PartitionModelUrl {
                    urls: vec![
                        "https://huggingface.co/part1".to_string(),
                        "https://modelscope.co/part1".to_string(),
                    ],
                    sha256: "test_sha256_1".to_string(),
                },
                PartitionModelUrl {
                    urls: vec![
                        "https://huggingface.co/part2".to_string(),
                        "https://modelscope.co/part2".to_string(),
                    ],
                    sha256: "test_sha256_2".to_string(),
                },
            ]),
        };
        let urls = super::filter_download_address(&model_info);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0].0, "https://huggingface.co/part1");
        assert_eq!(urls[0].1, "test_sha256_1");
        assert_eq!(urls[1].0, "https://huggingface.co/part2");
        assert_eq!(urls[1].1, "test_sha256_2");
    }

    #[test]
    fn test_filter_download_address_single_partition() {
        let model_info = ModelInfo {
            name: "test".to_string(),
            urls: None,
            sha256: Some("test_sha256".to_string()),
            prompt_template: None,
            chat_template: None,
            partition_urls: Some(vec![PartitionModelUrl {
                urls: vec!["https://huggingface.co/part1".to_string()],
                sha256: "test_sha256_1".to_string(),
            }]),
        };
        let urls = super::filter_download_address(&model_info);
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].0, "https://huggingface.co/part1");
        assert_eq!(urls[0].1, "test_sha256_1");
    }

    #[test]
    fn test_filter_download_address_prefer_urls() {
        let model_info = ModelInfo {
            name: "test".to_string(),
            urls: Some(vec!["https://huggingface.co/test".to_string()]),
            sha256: Some("test_sha256".to_string()),
            prompt_template: None,
            chat_template: None,
            partition_urls: Some(vec![PartitionModelUrl {
                urls: vec!["https://modelscope.co/test".to_string()],
                sha256: "test_sha256".to_string(),
            }]),
        };
        let urls = super::filter_download_address(&model_info);
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].0, "https://huggingface.co/test");
        assert_eq!(urls[0].1, "test_sha256");
    }

    #[test]
    fn test_filter_download_address_huggingface_override_urls() {
        std::env::set_var("TABBY_HUGGINGFACE_HOST_OVERRIDE", "modelscope.co");
        let model_info = ModelInfo {
            name: "test".to_string(),
            urls: Some(vec!["https://huggingface.co/test".to_string()]),
            sha256: Some("test_sha256".to_string()),
            prompt_template: None,
            chat_template: None,
            partition_urls: None,
        };
        let urls = super::filter_download_address(&model_info);
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].0, "https://modelscope.co/test");
        assert_eq!(urls[0].1, "test_sha256");
        // must reset the env, or it will affect other tests
        std::env::remove_var("TABBY_HUGGINGFACE_HOST_OVERRIDE");
    }

    #[test]
    fn test_filter_download_address_huggingface_override_partitioned() {
        std::env::set_var("TABBY_HUGGINGFACE_HOST_OVERRIDE", "modelscope.co");
        let model_info = ModelInfo {
            name: "test".to_string(),
            urls: None,
            sha256: Some("test_sha256".to_string()),
            prompt_template: None,
            chat_template: None,
            partition_urls: Some(vec![
                PartitionModelUrl {
                    urls: vec!["https://huggingface.co/part1".to_string()],
                    sha256: "test_sha256_1".to_string(),
                },
                PartitionModelUrl {
                    urls: vec!["https://huggingface.co/part2".to_string()],
                    sha256: "test_sha256_2".to_string(),
                },
            ]),
        };
        let urls = super::filter_download_address(&model_info);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0].0, "https://modelscope.co/part1");
        assert_eq!(urls[0].1, "test_sha256_1");
        assert_eq!(urls[1].0, "https://modelscope.co/part2");
        assert_eq!(urls[1].1, "test_sha256_2");
        // must reset the env, or it will affect other tests
        std::env::remove_var("TABBY_HUGGINGFACE_HOST_OVERRIDE");
    }

    #[test]
    fn test_filter_download_address_download_host() {
        std::env::set_var("TABBY_DOWNLOAD_HOST", "modelscope.co");
        let model_info = ModelInfo {
            name: "test".to_string(),
            urls: None,
            sha256: Some("test_sha256".to_string()),
            prompt_template: None,
            chat_template: None,
            partition_urls: Some(vec![
                PartitionModelUrl {
                    urls: vec![
                        "https://huggingface.co/part1".to_string(),
                        "https://modelscope.co/part1".to_string(),
                    ],
                    sha256: "test_sha256_1".to_string(),
                },
                PartitionModelUrl {
                    urls: vec![
                        "https://huggingface.co/part2".to_string(),
                        "https://modelscope.co/part2".to_string(),
                    ],
                    sha256: "test_sha256_2".to_string(),
                },
            ]),
        };
        let urls = super::filter_download_address(&model_info);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0].0, "https://modelscope.co/part1");
        assert_eq!(urls[0].1, "test_sha256_1");
        assert_eq!(urls[1].0, "https://modelscope.co/part2");
        assert_eq!(urls[1].1, "test_sha256_2");
        // must reset the env, or it will affect other tests
        std::env::remove_var("TABBY_DOWNLOAD_HOST");
    }
}
