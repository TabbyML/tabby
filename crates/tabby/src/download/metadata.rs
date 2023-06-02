use anyhow::Result;
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
    async fn from(model_id: &str) -> HFMetadata {
        let api_url = format!("https://huggingface.co/api/models/{}", model_id);
        reqwest::get(&api_url)
            .await
            .unwrap_or_else(|_| panic!("Failed to GET url '{}'", api_url))
            .json::<HFMetadata>()
            .await
            .unwrap_or_else(|_| panic!("Failed to parse HFMetadata '{}'", api_url))
    }
}

#[derive(Serialize, Deserialize)]
pub struct Metadata {
    auto_model: String,
    etags: HashMap<String, String>,
}

impl Metadata {
    pub async fn from(model_id: &str) -> Metadata {
        if let Some(metadata) = Self::from_local(model_id) {
            metadata
        } else {
            let hf_metadata = HFMetadata::from(model_id).await;
            Metadata {
                auto_model: hf_metadata.transformers_info.auto_model,
                etags: HashMap::new(),
            }
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

    pub fn local_cache_key(&self, path: &str) -> Option<&str> {
        self.etags.get(path).map(|x| x.as_str())
    }

    pub fn remote_cache_key(res: &reqwest::Response) -> &str {
        res.headers()
            .get("etag")
            .unwrap_or_else(|| panic!("Failed to GET ETAG header from '{}'", res.url()))
            .to_str()
            .unwrap_or_else(|_| panic!("Failed to convert ETAG header into string '{}'", res.url()))
    }

    pub async fn set_local_cache_key(&mut self, path: &str, cache_key: &str) {
        self.etags.insert(path.to_string(), cache_key.to_string());
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
        let hf_metadata = HFMetadata::from("TabbyML/J-350M").await;
        assert_eq!(
            hf_metadata.transformers_info.auto_model,
            "AutoModelForCausalLM"
        );
    }
}
