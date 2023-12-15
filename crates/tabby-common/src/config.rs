use std::{
    io::{Error, ErrorKind},
    path::PathBuf,
};

use filenamify::filenamify;
use serde::{Deserialize, Serialize};

use crate::path::{config_file, repositories_dir};

#[derive(Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub repositories: Vec<RepositoryConfig>,

    #[serde(default)]
    pub server: ServerConfig,
}

impl Config {
    pub fn load() -> Result<Self, Error> {
        let file = serdeconv::from_toml_file(crate::path::config_file().as_path());
        file.map_err(|err| {
            Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Config {:?} doesn't exist or is not valid: `{:?}`",
                    config_file(),
                    err
                ),
            )
        })
    }

    #[cfg(feature = "testutils")]
    pub fn save(&self) {
        serdeconv::to_toml_file(self, crate::path::config_file().as_path())
            .expect("Failed to write config file");
    }
}

#[derive(Serialize, Deserialize)]
pub struct RepositoryConfig {
    pub git_url: String,
}

impl RepositoryConfig {
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
        filenamify(&self.git_url)
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
        assert_eq!(repo.name(), "https_github.com_TabbyML_tabby.git");
    }
}
