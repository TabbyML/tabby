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
    pub repositories: Vec<Repository>,

    #[serde(default)]
    pub swagger: SwaggerConfig,
}

#[derive(Serialize, Deserialize, Default)]
pub struct SwaggerConfig {}

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
pub struct Repository {
    pub git_url: String,
}

impl Repository {
    pub fn dir(&self) -> PathBuf {
        if self.is_local_dir() {
            let path = self.git_url.strip_prefix("file://").unwrap();
            path.into()
        } else {
            repositories_dir().join(filenamify(&self.git_url))
        }
    }

    pub fn is_local_dir(&self) -> bool {
        self.git_url.starts_with("file://")
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, Repository};

    #[test]
    fn it_parses_empty_config() {
        let config = serdeconv::from_toml_str::<Config>("");
        debug_assert!(config.is_ok(), "{}", config.err().unwrap());
    }

    #[test]
    fn it_parses_local_dir() {
        let repo = Repository {
            git_url: "file:///home/user".to_owned(),
        };
        assert!(repo.is_local_dir());
        assert_eq!(repo.dir().display().to_string(), "/home/user");

        let repo = Repository {
            git_url: "https://github.com/TabbyML/tabby".to_owned(),
        };
        assert!(!repo.is_local_dir());
    }
}
