mod cache_info;
mod registry;

use std::{cmp, fs, io::Write, path::Path};

use anyhow::{anyhow, Result};
use cache_info::CacheInfo;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use registry::{create_registry, Registry};
use tabby_common::path::ModelDir;
use tokio_retry::{
    strategy::{jitter, ExponentialBackoff},
    Retry,
};

pub struct Downloader {
    model_id: String,
    prefer_local_file: bool,
    registry: Box<dyn Registry>,
}

impl Downloader {
    pub fn new(model_id: &str, prefer_local_file: bool) -> Self {
        Self {
            model_id: model_id.to_owned(),
            prefer_local_file,
            registry: create_registry(),
        }
    }

    #[deprecated]
    pub async fn download_ctranslate2_files(&self) -> Result<()> {
        let files = vec![
            ("tabby.json", true),
            ("tokenizer.json", true),
            ("ctranslate2/vocabulary.txt", false),
            ("ctranslate2/shared_vocabulary.txt", false),
            ("ctranslate2/vocabulary.json", false),
            ("ctranslate2/shared_vocabulary.json", false),
            ("ctranslate2/config.json", true),
            ("ctranslate2/model.bin", true),
        ];

        self.download_files(&files).await
    }

    pub async fn download_ggml_files(&self) -> Result<()> {
        let files = vec![("tabby.json", true), ("ggml/q8_0.v2.gguf", true)];
        self.download_files(&files).await
    }

    async fn download_files(&self, files: &[(&str, bool)]) -> Result<()> {
        // Local path, no need for downloading.
        if fs::metadata(&self.model_id).is_ok() {
            return Ok(());
        }

        let mut cache_info = CacheInfo::from(&self.model_id).await;
        for (path, required) in files {
            download_model_file(
                self.registry.as_ref(),
                &mut cache_info,
                &self.model_id,
                path,
                self.prefer_local_file,
                *required,
            )
            .await?;
        }
        Ok(())
    }
}

async fn download_model_file(
    registry: &dyn Registry,
    cache_info: &mut CacheInfo,
    model_id: &str,
    path: &str,
    prefer_local_file: bool,
    required: bool,
) -> Result<()> {
    // Create url.
    let url = registry.build_url(model_id, path);

    // Create destination path.
    let filepath = ModelDir::new(model_id).path_string(path);

    // Get cache key.
    let local_cache_key = cache_info.local_cache_key(path);

    // Check local file ready.
    let local_cache_key = local_cache_key
        // local cache key is only valid if == 404 or local file exists.
        // FIXME(meng): use sha256 to validate file is ready.
        .filter(|&local_cache_key| local_cache_key == "404" || fs::metadata(&filepath).is_ok());

    let strategy = ExponentialBackoff::from_millis(100).map(jitter).take(2);
    let download_job = Retry::spawn(strategy, || {
        download_file(registry, &url, &filepath, local_cache_key, !required)
    });
    if let Ok(etag) = download_job.await {
        cache_info.set_local_cache_key(path, &etag).await;
    } else if prefer_local_file && local_cache_key.is_some() {
        // Do nothing.
    } else {
        return Err(anyhow!("Failed to fetch url {}", url));
    }

    cache_info.save(model_id)?;
    Ok(())
}

async fn download_file(
    registry: &dyn Registry,
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

    let remote_cache_key = registry.build_cache_key(url).await?;
    if local_cache_key == Some(remote_cache_key.as_str()) {
        return Ok(remote_cache_key);
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
