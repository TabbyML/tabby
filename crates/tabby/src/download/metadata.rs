use serde::{Serialize, Deserialize};
use error_chain::error_chain;
use std::collections::HashMap;
use std::fs;
use tabby_common::path::ModelDir;

error_chain! {
     foreign_links{
         HttpRequest(reqwest::Error);
         HeaderError(reqwest::header::ToStrError);
         JsonParseError(serdeconv::Error);
     }

     errors {
         PathNotExist(t: String) {
            description("Path doesn't exist")
            display("Path doesn't exist: '{}'", t)
         }

         EtagHeaderNotPresent(t: String) {
            description("Missing etag header")
            display("Missing etag header: '{}'", t)
         }
     }
}

#[derive(Deserialize)]
struct HFTransformersInfo {
    auto_model: String
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct HFMetadata {
    transformers_info: HFTransformersInfo
}

impl HFMetadata {
    async fn from(model_id: &str) -> Result<HFMetadata> {
        let api_url = format!("https://huggingface.co/api/models/{}", model_id);
        let metadata = reqwest::get(api_url)
            .await?
            .json::<HFMetadata>()
            .await?;
        Ok(metadata)
    }
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    auto_model: String,
    etags: HashMap<String, String>
}

impl Metadata {
    pub async fn new(model_id: &str) -> Result<Metadata> {
        let hf_metadata = HFMetadata::from(model_id).await?;
        let metadata = Metadata {
            auto_model: hf_metadata.transformers_info.auto_model,
            etags: HashMap::new()
        };
        Ok(metadata)
    }

    pub async fn from(model_id: &str) -> Result<Metadata> {
        let metadata_file = ModelDir::new(model_id).metadata_file();
        if !fs::metadata(&metadata_file).is_ok() {
            Err(ErrorKind::PathNotExist(metadata_file).into())
        } else {
            let metadata : Metadata = serdeconv::from_json_file(metadata_file)?;
            Ok(metadata)
        }
    }

    pub async fn etag_matches(&self, url: &str, path: &str) -> Result<bool> {
        let etag = self.etags.get(path).ok_or(ErrorKind::PathNotExist(path.to_owned()))?;
        let etag_from_header = reqwest::get(url)
            .await?
            .headers()
            .get("etag").ok_or(ErrorKind::EtagHeaderNotPresent(url.to_owned()))?
            .to_str()?
            .to_owned();

        Ok(etag == &etag_from_header)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hf() {
        let hf_metadata = HFMetadata::from("TabbyML/J-350M").await.unwrap();
        assert_eq!(hf_metadata.transformers_info.auto_model, "AutoModelForCausalLM");
    }
}
