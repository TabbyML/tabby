use anyhow::{anyhow, Result};
use async_trait::async_trait;

use crate::Registry;

#[derive(Default)]
pub struct HuggingFaceRegistry {}

#[async_trait]
impl Registry for HuggingFaceRegistry {
    fn build_url(&self, model_id: &str, path: &str) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", model_id, path)
    }

    async fn build_cache_key(&self, url: &str) -> Result<String> {
        let res = reqwest::get(url).await?;
        let cache_key = res
            .headers()
            .get("etag")
            .ok_or(anyhow!("etag key missing"))?
            .to_str()?;
        Ok(cache_key.to_owned())
    }
}
