use std::{collections::HashSet, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use derive_builder::Builder;
use hash_ids::HashIds;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{
    path::repositories_dir,
    terminal::{HeaderFormat, InfoMessage},
};

#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct Config {
    #[serde(default)]
    pub repositories: Vec<RepositoryConfig>,

    #[serde(default)]
    pub server: ServerConfig,

    #[serde(default)]
    pub model: ModelConfigGroup,
}

impl Config {
    pub fn load() -> Result<Self> {
        let cfg_path = crate::path::config_file();
        if !cfg_path.as_path().exists() {
            debug!(
                "Config file {} not found, apply default configuration",
                cfg_path.display()
            );
            return Ok(Default::default());
        }
        let mut cfg: Self = serdeconv::from_toml_file(cfg_path.as_path())
            .context(format!("Config file '{}' is not valid", cfg_path.display()))?;

        if let Err(e) = cfg.validate_dirs() {
            cfg = Default::default();
            InfoMessage::new(
                "Parsing config failed",
                HeaderFormat::BoldRed,
                &[
                    &format!(
                        "Warning: Could not parse the Tabby configuration at {}",
                        crate::path::config_file().as_path().to_string_lossy()
                    ),
                    &format!("Reason: {e}"),
                    "Falling back to default config, please resolve the errors and restart Tabby",
                ],
            )
            .print();
        }

        Ok(cfg)
    }

    #[cfg(feature = "testutils")]
    pub fn save(&self) {
        serdeconv::to_toml_file(self, crate::path::config_file().as_path())
            .expect("Failed to write config file");
    }

    fn validate_dirs(&self) -> Result<()> {
        let mut dirs = HashSet::new();
        for repo in self.repositories.iter() {
            let dir = repo.dir().display().to_string();
            if !dirs.insert(dir.clone()) {
                return Err(anyhow!("Duplicate directory in `repositories`: {}", dir));
            }
        }
        Ok(())
    }
}

lazy_static! {
    static ref HASHER: HashIds = HashIds::builder()
        .with_salt("tabby-config-id-serializer")
        .with_min_length(6)
        .finish();
}

pub fn config_index_to_id(index: usize) -> String {
    let id = HASHER.encode(&[index as u64]);
    format!("config:{id}")
}

pub fn config_id_to_index(id: &str) -> Result<usize, anyhow::Error> {
    let id = id
        .strip_prefix("config:")
        .ok_or_else(|| anyhow!("Invalid config ID"))?;

    HASHER
        .decode(id)
        .first()
        .map(|i| *i as usize)
        .ok_or_else(|| anyhow!("Invalid config ID"))
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RepositoryConfig {
    pub git_url: String,
}

impl RepositoryConfig {
    pub fn new<T: Into<String>>(git_url: T) -> Self {
        Self {
            git_url: git_url.into(),
        }
    }

    pub fn canonical_git_url(&self) -> String {
        Self::canonicalize_url(&self.git_url)
    }

    pub fn canonicalize_url(url: &str) -> String {
        let url = url.strip_suffix(".git").unwrap_or(url);
        url::Url::parse(url)
            .map(|mut url| {
                let _ = url.set_password(None);
                let _ = url.set_username("");
                url.to_string()
            })
            .unwrap_or_else(|_| url.to_string())
    }

    pub fn dir(&self) -> PathBuf {
        if self.is_local_dir() {
            let path = self.git_url.strip_prefix("file://").unwrap();
            path.into()
        } else {
            repositories_dir().join(self.dir_name())
        }
    }

    pub fn dir_name(&self) -> String {
        sanitize_name(&self.canonical_git_url())
    }

    pub fn is_local_dir(&self) -> bool {
        self.git_url.starts_with("file://")
    }
}

fn sanitize_name(s: &str) -> String {
    let mut sanitized: Vec<char> = s
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '.' | '-' => c,
            _ => '_',
        })
        .collect();
    sanitized.dedup_by(|a, b| *a == '_' && *b == '_');
    sanitized.into_iter().collect()
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ServerConfig {
    /// The timeout in seconds for the /v1/completion api.
    pub completion_timeout: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            completion_timeout: 30,
        }
    }
}

fn default_embedding_config() -> ModelConfig {
    ModelConfig::Local(LocalModelConfig {
        model_id: "Nomic-Embed-Text".into(),
        parallelism: 1,
        num_gpu_layers: 9999,
        enable_fast_attention: false,
    })
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelConfigGroup {
    pub completion: Option<ModelConfig>,
    pub chat: Option<ModelConfig>,
    #[serde(default = "default_embedding_config")]
    pub embedding: ModelConfig,
}

impl Default for ModelConfigGroup {
    fn default() -> Self {
        Self {
            completion: None,
            chat: None,
            embedding: default_embedding_config(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ModelConfig {
    Http(HttpModelConfig),
    Local(LocalModelConfig),
}

#[derive(Serialize, Deserialize, Builder, Debug, Clone)]
pub struct HttpModelConfig {
    /// The kind of model, we have three group of models:
    /// 1. Completion API [CompletionStream](tabby_inference::CompletionStream)
    ///   - llama.cpp/completion: llama.cpp `/completion` API.
    /// 2. Chat API: [ChatCompletionStream](tabby_inference::ChatCompletionStream)
    ///   - openai-chat: OpenAI /v1/chat/completions API.
    /// 3. Embedding API [Embedding](tabby_inference::Embedding)
    ///   - llama.cpp/embedding: llama.cpp `/embedding` API.
    pub kind: String,

    pub api_endpoint: String,

    #[builder(default)]
    pub api_key: Option<String>,

    /// Used by OpenAI style API for model name.
    #[builder(default)]
    pub model_name: Option<String>,

    /// Used by Completion API to construct a completion model.
    #[builder(default)]
    pub prompt_template: Option<String>,

    /// Used by Completion API to construct a chat model.
    #[builder(default)]
    pub chat_template: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LocalModelConfig {
    pub model_id: String,

    #[serde(default = "default_parallelism")]
    pub parallelism: u8,

    #[serde(default = "default_num_gpu_layers")]
    pub num_gpu_layers: u16,

    #[serde(default)]
    pub enable_fast_attention: bool,
}

fn default_parallelism() -> u8 {
    1
}

fn default_num_gpu_layers() -> u16 {
    9999
}

#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct DocIndexConfig {
    pub start_urls: Vec<String>,
}

#[async_trait]
pub trait ConfigAccess: Send + Sync {
    async fn repositories(&self) -> Result<Vec<RepositoryConfig>>;
}

pub struct StaticConfigAccess;

#[async_trait]
impl ConfigAccess for StaticConfigAccess {
    async fn repositories(&self) -> Result<Vec<RepositoryConfig>> {
        Ok(Config::load()?.repositories)
    }
}

#[cfg(test)]
mod tests {
    use super::{sanitize_name, Config, RepositoryConfig};

    #[test]
    fn it_parses_empty_config() {
        let config = serdeconv::from_toml_str::<Config>("");
        debug_assert!(config.is_ok(), "{}", config.err().unwrap());
    }

    #[test]
    fn it_parses_local_dir() {
        let repo = RepositoryConfig {
            git_url: "file:///home/user".to_owned(),
        };
        assert!(repo.is_local_dir());
        assert_eq!(repo.dir().display().to_string(), "/home/user");

        let repo = RepositoryConfig {
            git_url: "https://github.com/TabbyML/tabby".to_owned(),
        };
        assert!(!repo.is_local_dir());
    }

    #[test]
    fn test_repository_config_name() {
        let repo = RepositoryConfig {
            git_url: "https://github.com/TabbyML/tabby.git".to_owned(),
        };
        assert!(repo.dir().ends_with("https_github.com_TabbyML_tabby"));
    }

    #[test]
    fn test_sanitize_repository_name() {
        assert_eq!(sanitize_name("abc@def"), "abc_def");
        assert_eq!(sanitize_name("abcdef"), "abcdef");
        assert_eq!(
            sanitize_name("https://github.com/TabbyML/tabby.git"),
            "https_github.com_TabbyML_tabby.git"
        );
    }

    #[test]
    fn test_canonicalize_url() {
        assert_eq!(
            RepositoryConfig::canonicalize_url("https://abc:dev@github.com/"),
            "https://github.com/"
        );

        assert_eq!(
            RepositoryConfig::canonicalize_url("https://token@github.com/TabbyML/tabby"),
            "https://github.com/TabbyML/tabby"
        );

        assert_eq!(
            RepositoryConfig::canonicalize_url("https://github.com/TabbyML/tabby"),
            "https://github.com/TabbyML/tabby"
        );

        assert_eq!(
            RepositoryConfig::canonicalize_url("https://github.com/TabbyML/tabby.git"),
            "https://github.com/TabbyML/tabby"
        );

        assert_eq!(
            RepositoryConfig::canonicalize_url("file:///home/TabbyML/tabby"),
            "file:///home/TabbyML/tabby"
        );
    }
}
