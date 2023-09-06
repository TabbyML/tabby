mod cache_info;

use std::{cmp, fs, io::Write, path::Path};

use anyhow::{anyhow, Result};
use cache_info::CacheInfo;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use tabby_common::path::ModelDir;
use tokio_retry::{
    strategy::{jitter, ExponentialBackoff},
    Retry,
};
use tracing::info;

impl CacheInfo {
    async fn download(
        &mut self,
        model_id: &str,
        path: &str,
        prefer_local_file: bool,
        is_optional: bool,
    ) -> Result<()> {
        // Create url.
        let url = format!("https://huggingface.co/{}/resolve/main/{}", model_id, path);

        // Get cache key.
        let local_cache_key = self.local_cache_key(path);

        // Create destination path.
        let filepath = ModelDir::new(model_id).path_string(path);

        // Cache hit.
        let local_file_ready = if prefer_local_file {
            if let Some(local_cache_key) = local_cache_key {
                if local_cache_key == "404" {
                    true
                } else {
                    fs::metadata(&filepath).is_ok()
                }
            } else {
                false
            }
        } else {
            false
        };

        if !local_file_ready {
            let strategy = ExponentialBackoff::from_millis(100).map(jitter).take(3);
            let etag = Retry::spawn(strategy, || {
                download_file(&url, &filepath, local_cache_key, is_optional)
            })
            .await?;
            self.set_local_cache_key(path, &etag).await;
        }
        Ok(())
    }
}

pub async fn download_model(model_id: &str, prefer_local_file: bool) -> Result<()> {
    if fs::metadata(model_id).is_ok() {
        // Local path, no need for downloading.
        return Ok(());
    }

    info!("Start downloading model `{}`", model_id);

    let mut cache_info = CacheInfo::from(model_id).await;

    let optional_files = vec![
        "ctranslate2/vocabulary.txt",
        "ctranslate2/shared_vocabulary.txt",
        "ctranslate2/vocabulary.json",
        "ctranslate2/shared_vocabulary.json",
        "ggml/q8_0.gguf"
    ];
    for path in optional_files {
        cache_info
            .download(
                model_id,
                path,
                prefer_local_file,
                /* is_optional */ true,
            )
            .await?;
    }

    let required_files = vec![
        "tabby.json",
        "tokenizer.json",
        "ctranslate2/config.json",
        "ctranslate2/model.bin",
    ];
    for path in required_files {
        cache_info
            .download(
                model_id,
                path,
                prefer_local_file,
                /* required= */ false,
            )
            .await?;
    }

    cache_info.save(model_id)?;
    Ok(())
}

async fn download_file(
    url: &str,
    path: &str,
    local_cache_key: Option<&str>,
    is_optional: bool,
) -> Result<String> {
    fs::create_dir_all(Path::new(path).parent().unwrap())?;

    // Reqwest setup
    let res = reqwest::get(url).await?;

    if is_optional && res.status() == 404 {
        // Cache 404 for optional file.
        return Ok("404".to_owned());
    }

    if !res.status().is_success() {
        return Err(anyhow!(format!("Invalid url: {}", url)));
    }

    let remote_cache_key = CacheInfo::remote_cache_key(&res)?.to_string();
    if let Some(local_cache_key) = local_cache_key {
        if local_cache_key == remote_cache_key {
            return Ok(remote_cache_key);
        }
    }

    let total_size = res
        .content_length()
        .ok_or(anyhow!("No content length in headers"))?;

    // Indicatif setup
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
        .progress_chars("#>-"));
    pb.set_message(format!("Downloading {}", path));

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

    pb.finish_with_message(format!("Downloaded {}", path));
    Ok(remote_cache_key)
}
