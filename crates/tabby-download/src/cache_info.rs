use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tabby_common::path::ModelDir;

#[derive(Serialize, Deserialize)]
pub struct CacheInfo {
    etags: HashMap<String, String>,
}

impl CacheInfo {
    pub async fn from(model_id: &str) -> CacheInfo {
        if let Some(cache_info) = Self::from_local(model_id) {
            cache_info
        } else {
            CacheInfo {
                etags: HashMap::new(),
            }
        }
    }

    fn from_local(model_id: &str) -> Option<CacheInfo> {
        let cache_info_file = ModelDir::new(model_id).cache_info_file();
        if fs::metadata(&cache_info_file).is_ok() {
            serdeconv::from_json_file(cache_info_file).ok()
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
        let cache_info_file = ModelDir::new(model_id).cache_info_file();
        let cache_info_file_path = Path::new(&cache_info_file);
        serdeconv::to_json_file(self, cache_info_file_path)?;
        Ok(())
    }
}
