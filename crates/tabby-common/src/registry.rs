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
    pub sha256: String,
}

fn models_json_file(registry: &str) -> PathBuf {
    models_dir().join(registry).join("models.json")
}

async fn load_remote_registry(registry: &str) -> Result<Vec<ModelInfo>> {
    let model_info = reqwest::get(format!(
        "https://raw.githubusercontent.com/{}/registry-tabby/main/models.json",
        registry
    ))
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

impl ModelRegistry {
    pub async fn new(registry: &str) -> Self {
        Self {
            name: registry.to_owned(),
            models: load_remote_registry(registry).await.unwrap_or_else(|err| {
                load_local_registry(registry).unwrap_or_else(|_| {
                    panic!(
                        "Failed to fetch model organization <{}>: {:?}",
                        registry, err
                    )
                })
            }),
        }
    }

    fn get_model_dir(&self, name: &str) -> PathBuf {
        models_dir().join(&self.name).join(name)
    }

    pub fn migrate_model_path(&self, name: &str) -> Result<(), std::io::Error> {
        let model_path = self.get_model_path(name);
        let old_model_path = self
            .get_model_dir(name)
            .join(LEGACY_GGML_MODEL_RELATIVE_PATH.as_str());

        if !model_path.exists() && old_model_path.exists() {
            std::fs::rename(&old_model_path, &model_path)?;
            #[cfg(target_family = "unix")]
            std::os::unix::fs::symlink(&model_path, &old_model_path)?;
            #[cfg(target_family = "windows")]
            std::os::windows::fs::symlink_file(&model_path, &old_model_path)?;
        }
        Ok(())
    }

    pub fn get_model_path(&self, name: &str) -> PathBuf {
        self.get_model_dir(name)
            .join(GGML_MODEL_RELATIVE_PATH.as_str())
    }

    pub fn save_model_info(&self, name: &str) {
        let model_info = self.get_model_info(name);
        let path = self.get_model_dir(name).join("tabby.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        serdeconv::to_json_file(model_info, path).unwrap();
    }

    pub fn get_model_info(&self, name: &str) -> &ModelInfo {
        self.models
            .iter()
            .find(|x| x.name == name)
            .unwrap_or_else(|| panic!("Invalid model_id <{}/{}>", self.name, name))
    }
}

pub fn parse_model_id(model_id: &str) -> (&str, &str) {
    let parts: Vec<_> = model_id.split('/').collect();
    if parts.len() == 1 {
        ("TabbyML", parts[0])
    } else if parts.len() == 2 {
        (parts[0], parts[1])
    } else {
        panic!("Invalid model id {}", model_id);
    }
}

lazy_static! {
    pub static ref LEGACY_GGML_MODEL_RELATIVE_PATH: String =
        format!("ggml{}q8_0.v2.gguf", std::path::MAIN_SEPARATOR_STR);
    pub static ref GGML_MODEL_RELATIVE_PATH: String =
        format!("ggml{}model.gguf", std::path::MAIN_SEPARATOR_STR);
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

        let old_model_path = dir.join(LEGACY_GGML_MODEL_RELATIVE_PATH.as_str());
        tokio::fs::create_dir_all(old_model_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&old_model_path)
            .await
            .unwrap();

        registry.migrate_model_path("StarCoder-1B").unwrap();
        assert!(registry.get_model_path("StarCoder-1B").exists());
        assert!(old_model_path.exists());
    }
}
