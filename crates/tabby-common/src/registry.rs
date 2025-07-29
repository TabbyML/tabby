use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

use crate::path::models_dir;

#[derive(Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub urls: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    // partition_urls is used for model download address
    // if the model is partitioned, the addresses of each partition will be listed here,
    // if there is only one partition, it will be the same as `urls`.
    //
    // will first try to the `urls`, if not found, will try this `partition_urls`.
    //
    // must make sure the first address is the entrypoint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partition_urls: Option<Vec<PartitionModelUrl>>,
}

#[derive(Serialize, Deserialize)]
pub struct PartitionModelUrl {
    pub urls: Vec<String>,
    pub sha256: String,
}

fn models_json_file(registry: &str) -> PathBuf {
    models_dir().join(registry).join("models.json")
}

async fn load_remote_registry(registry: &str) -> Result<Vec<ModelInfo>> {
    // Create an HTTP client with a custom timeout.
    // This is necessary because the default timeout settings can sometimes cause requests to hang indefinitely.
    // To prevent such issues, we specify a custom timeout duration.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .context("Failed to build HTTP client")?;

    let model_info = client
        .get(format!(
            "https://raw.githubusercontent.com/{registry}/registry-tabby/main/models.json"
        ))
        .send()
        .await
        .context("Failed to download")?
        .json()
        .await
        .context("Failed to get JSON")?;

    let dir = models_dir().join(registry);
    // We don't want to fail if the TabbyML directory already exists,
    // which is exactly, what `create_dir_all` will do, see
    // https://doc.rust-lang.org/std/fs/fn.create_dir.html#errors.
    if !dir.exists() {
        fs::create_dir_all(&dir).context(format!("Failed to create dir {dir:?}"))?;
    }
    serdeconv::to_json_file(&model_info, models_json_file(registry))
        .context("Failed to convert JSON to file")?;
    Ok(model_info)
}

fn load_local_registry(registry: &str) -> Result<Vec<ModelInfo>> {
    Ok(serdeconv::from_json_file(models_json_file(registry))?)
}

#[derive(Default)]
pub struct ModelRegistry {
    pub name: String,
    pub models: Vec<ModelInfo>,
}

lazy_static! {
    pub static ref LEGACY_GGML_MODEL_PATH: String =
        format!("ggml{}model.gguf", std::path::MAIN_SEPARATOR_STR);
    pub static ref GGML_MODEL_PARTITIONED_PREFIX: String = "model-00001-of-".into();
}

// model registry tree structure
// root: ~/.tabby/models/TabbyML
//
// fn get_model_root_dir(model_name) -> {root}/{model_name}
//
// fn get_model_dir(model_name) -> {root}/{model_name}/ggml
impl ModelRegistry {
    pub async fn new(registry: &str) -> Self {
        Self {
            name: registry.to_owned(),
            models: load_remote_registry(registry).await.unwrap_or_else(|err| {
                load_local_registry(registry).unwrap_or_else(|_| {
                    panic!("Failed to fetch model organization <{registry}>: {err:?}")
                })
            }),
        }
    }

    // get_model_store_dir returns {root}/{name}/ggml, e.g.. ~/.tabby/models/TabbyML/StarCoder-1B/ggml
    pub fn get_model_store_dir(&self, name: &str) -> PathBuf {
        self.get_model_dir(name).join("ggml")
    }

    // get_model_dir returns {root}/{name}, e.g. ~/.tabby/models/TabbyML/StarCoder-1B
    pub fn get_model_dir(&self, name: &str) -> PathBuf {
        models_dir().join(&self.name).join(name)
    }

    // get_model_path returns the entrypoint of the model,
    // will look for the file with the prefix "00001-of-"
    pub fn get_model_entry_path(&self, name: &str) -> Option<PathBuf> {
        for entry in fs::read_dir(self.get_model_store_dir(name)).ok()? {
            let entry = entry.expect("Error reading directory entry");
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();

            // Check if the file name starts with the specified prefix
            if file_name_str.starts_with(GGML_MODEL_PARTITIONED_PREFIX.as_str()) {
                return Some(entry.path()); // Return the full path as PathBuf
            }
        }

        None
    }

    pub fn migrate_legacy_model_path(&self, name: &str) -> Result<(), std::io::Error> {
        let old_model_path = self
            .get_model_dir(name)
            .join(LEGACY_GGML_MODEL_PATH.as_str());

        if old_model_path.exists() {
            return self.migrate_model_path(name, &old_model_path);
        }

        Ok(())
    }

    pub fn get_model_path(&self, name: &str) -> PathBuf {
        self.get_model_dir(name)
            .join(LEGACY_GGML_MODEL_PATH.as_str())
    }

    pub fn migrate_model_path(
        &self,
        name: &str,
        old_model_path: &PathBuf,
    ) -> Result<(), std::io::Error> {
        // legacy model always has a single file
        let model_path = self
            .get_model_store_dir(name)
            .join("model-00001-of-00001.gguf");
        std::fs::rename(old_model_path, model_path)?;
        Ok(())
    }

    pub fn save_model_info(&self, name: &str) {
        let model_info = self.get_model_info(name);
        let path = self.get_model_dir(name).join("tabby.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        serdeconv::to_json_file(model_info, path).unwrap();
    }

    pub fn get_model_info(&self, name: &str) -> &ModelInfo {
        match self.models.iter().find(|x| x.name == name) {
            Some(model_info) => model_info,
            None => panic!(
                "Invalid `model_id` <{}/{}>; please consult https://github.com/{}/registry-tabby for the correct `model_id`.",
                self.name, name, self.name
            ),
        }
    }
}

pub fn parse_model_id(model_id: &str) -> (&str, &str) {
    let parts: Vec<_> = model_id.split('/').collect();
    if parts.len() == 1 {
        ("TabbyML", parts[0])
    } else if parts.len() == 2 {
        (parts[0], parts[1])
    } else {
        panic!("Invalid model id {model_id}");
    }
}

#[cfg(test)]
mod tests {
    use temp_testdir::TempDir;

    use super::{ModelRegistry, *};
    use crate::path::set_tabby_root;

    #[tokio::test]
    async fn test_model_migration() {
        let root = TempDir::default();
        set_tabby_root(root.to_path_buf());

        let registry = ModelRegistry::new("TabbyML").await;
        let dir = registry.get_model_dir("StarCoder-1B");

        let old_model_path = dir.join(LEGACY_GGML_MODEL_PATH.as_str());
        let new_model_path = dir.join("ggml").join("model-00001-of-00001.gguf");
        tokio::fs::create_dir_all(old_model_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&old_model_path)
            .await
            .unwrap();

        assert!(!new_model_path.exists());
        registry.migrate_legacy_model_path("StarCoder-1B").unwrap();
        assert!(registry
            .get_model_entry_path("StarCoder-1B")
            .unwrap()
            .exists());
        assert!(!old_model_path.exists());
        assert!(new_model_path.exists());
    }
}
