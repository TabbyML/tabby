use std::{collections::HashSet, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    path::repositories_dir,
    terminal::{HeaderFormat, InfoMessage},
    SourceFile,
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
        let cfg_path = crate::path::config_file();
        let mut cfg: Self = serdeconv::from_toml_file(cfg_path.as_path()).context(format!(
            "Config file '{}' is missing or not valid",
            cfg_path.display()
        ))?;

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
                return Err(anyhow!("Duplicate dir in `repositories`: {}", dir));
            }
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RepositoryConfig {
    pub git_url: String,
}

impl RepositoryConfig {
    pub fn new(git_url: String) -> Self {
        Self { git_url }
    }

    pub fn dir(&self) -> PathBuf {
        if self.is_local_dir() {
            let path = self.git_url.strip_prefix("file://").unwrap();
            path.into()
        } else {
            repositories_dir().join(sanitize_name(&self.git_url))
        }
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
    fn start_snapshot(&self, _version: u64) {}
    fn process_file(&self, _version: u64, _file: SourceFile) {}
    fn finish_snapshot(&self, _version: u64) {}
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
        assert!(repo.dir().ends_with("https_github.com_TabbyML_tabby.git"));
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
}
