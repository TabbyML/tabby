use anyhow::{anyhow, Result};

#[derive(Default)]
pub struct HuggingFaceRegistry {}

impl HuggingFaceRegistry {
    pub fn build_url(&self, model_id: &str, path: &str) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", model_id, path)
    }

    pub async fn build_cache_key(&self, url: &str) -> Result<String> {
        let res = reqwest::get(url).await?;
        let cache_key = res
            .headers()
            .get("etag")
            .ok_or(anyhow!("etag key missing"))?
            .to_str()?;
        Ok(cache_key.to_owned())
    }
}
