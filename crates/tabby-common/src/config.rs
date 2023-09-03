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
    pub experimental: Experimental,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Experimental {
    #[serde(default = "default_as_false")]
    pub enable_prompt_rewrite: bool,
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
pub struct Repository {
    pub git_url: String,
}

impl Repository {
    pub fn dir(&self) -> PathBuf {
        repositories_dir().join(filenamify(&self.git_url))
    }
}

fn default_as_false() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn it_parses_empty_config() {
        let config = serdeconv::from_toml_str::<Config>("");
        debug_assert!(config.is_ok(), "{}", config.err().unwrap());
    }
}
