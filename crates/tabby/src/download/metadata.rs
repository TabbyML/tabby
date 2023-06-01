use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tabby_common::path::ModelDir;

#[derive(Deserialize)]
struct HFTransformersInfo {
    auto_model: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct HFMetadata {
    transformers_info: HFTransformersInfo,
}

impl HFMetadata {
    async fn from(model_id: &str) -> Result<HFMetadata> {
        let api_url = format!("https://huggingface.co/api/models/{}", model_id);
        let metadata = reqwest::get(api_url).await?.json::<HFMetadata>().await?;
        Ok(metadata)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Metadata {
    auto_model: String,
    etags: HashMap<String, String>,
}

impl Metadata {
    pub async fn from(model_id: &str) -> Result<Metadata> {
        if let Some(metadata) = Self::from_local(model_id) {
            Ok(metadata)
        } else {
            let hf_metadata = HFMetadata::from(model_id).await?;
            let metadata = Metadata {
                auto_model: hf_metadata.transformers_info.auto_model,
                etags: HashMap::new(),
            };
            Ok(metadata)
        }
    }

    fn from_local(model_id: &str) -> Option<Metadata> {
        let metadata_file = ModelDir::new(model_id).metadata_file();
        if fs::metadata(&metadata_file).is_ok() {
            let metadata = serdeconv::from_json_file(metadata_file);
            metadata.ok()
        } else {
            None
        }
    }

    pub fn has_etag(&self, url: &str) -> bool {
        self.etags.get(url).is_some()
    }

    pub async fn match_etag(&self, url: &str, path: &str) -> Result<bool> {
        let etag = self
            .etags
            .get(url)
            .ok_or(anyhow!("Path doesn't exist: {}", path))?;
        let etag_from_header = reqwest::get(url)
            .await?
            .headers()
            .get("etag")
            .ok_or(anyhow!("URL doesn't have etag header: '{}'", url))?
            .to_str()?
            .to_owned();

        Ok(etag == &etag_from_header)
    }

    pub async fn update_etag(&mut self, url: &str, path: &str) {
        self.etags.insert(url.to_owned(), path.to_owned());
    }

    pub fn save(&self, model_id: &str) -> Result<()> {
        let metadata_file = ModelDir::new(model_id).metadata_file();
        let metadata_file_path = Path::new(&metadata_file);
        serdeconv::to_json_file(self, metadata_file_path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hf() {
        let hf_metadata = HFMetadata::from("TabbyML/J-350M").await.unwrap();
        assert_eq!(
            hf_metadata.transformers_info.auto_model,
            "AutoModelForCausalLM"
        );
    }
}
