use std::{collections::HashSet, path::PathBuf};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use filenamify::filenamify;
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{
    path::repositories_dir,
    terminal::{HeaderFormat, InfoMessage},
};

#[derive(Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub repositories: Vec<RepositoryConfig>,

    #[serde(default)]
    pub server: ServerConfig,
}

impl Config {
    pub fn load() -> Result<Self> {
        let mut cfg: Self = serdeconv::from_toml_file(crate::path::config_file().as_path())?;

        if let Err(e) = cfg.validate_names() {
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

    fn validate_names(&self) -> Result<()> {
        let mut names = HashSet::new();
        for repo in self.repositories.iter() {
            let name = repo.name();
            if !RepositoryConfig::validate_name(&name) {
                return Err(anyhow!("Invalid characters in repository name: {}", name));
            }
            if !names.insert(repo.name()) {
                return Err(anyhow!("Duplicate name in `repositories`: {}", repo.name()));
            }
        }
        Ok(())
    }
}

lazy_static! {
    pub static ref REPOSITORY_NAME_REGEX: Regex = Regex::new("[a-zA-Z][a-zA-Z0-9-]+").unwrap();
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RepositoryConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    pub git_url: String,
}

impl RepositoryConfig {
    #[cfg(feature = "testutils")]
    pub fn new(git_url: String) -> Self {
        Self {
            name: None,
            git_url,
        }
    }

    pub fn new_named(name: String, git_url: String) -> Self {
        Self {
            name: Some(name),
            git_url,
        }
    }

    pub fn validate_name(name: &str) -> bool {
        REPOSITORY_NAME_REGEX.is_match(name)
    }

    pub fn dir(&self) -> PathBuf {
        if self.is_local_dir() {
            let path = self.git_url.strip_prefix("file://").unwrap();
            path.into()
        } else {
            repositories_dir().join(self.name())
        }
    }

    pub fn is_local_dir(&self) -> bool {
        self.git_url.starts_with("file://")
    }

    pub fn name(&self) -> String {
        if let Some(name) = &self.name {
            name.clone()
        } else {
            filenamify(&self.git_url)
        }
    }
}

#[derive(Serialize, Deserialize)]
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

#[async_trait]
pub trait RepositoryAccess: Send + Sync {
    async fn list_repositories(&self) -> Result<Vec<RepositoryConfig>>;
}

pub struct ConfigRepositoryAccess;

#[async_trait]
impl RepositoryAccess for ConfigRepositoryAccess {
    async fn list_repositories(&self) -> Result<Vec<RepositoryConfig>> {
        Ok(Config::load()?.repositories)
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, RepositoryConfig};

    #[test]
    fn it_parses_empty_config() {
        let config = serdeconv::from_toml_str::<Config>("");
        debug_assert!(config.is_ok(), "{}", config.err().unwrap());
    }

    #[test]
    fn it_parses_local_dir() {
        let repo = RepositoryConfig {
            name: None,
            git_url: "file:///home/user".to_owned(),
        };
        assert!(repo.is_local_dir());
        assert_eq!(repo.dir().display().to_string(), "/home/user");

        let repo = RepositoryConfig {
            name: None,
            git_url: "https://github.com/TabbyML/tabby".to_owned(),
        };
        assert!(!repo.is_local_dir());
    }

    #[test]
    fn test_repository_config_name() {
        let repo = RepositoryConfig {
            name: None,
            git_url: "https://github.com/TabbyML/tabby.git".to_owned(),
        };
        assert_eq!(repo.name(), "https_github.com_TabbyML_tabby.git");
    }
}
