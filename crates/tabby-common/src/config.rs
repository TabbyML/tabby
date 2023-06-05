use filenamify::filenamify;
use std::path::PathBuf;

use serde::Deserialize;

use crate::path::repositories_dir;

#[derive(Deserialize)]
pub struct Config {
    pub repositories: Vec<Repository>,
}

impl Config {
    pub fn load() -> Self {
        serdeconv::from_toml_file(crate::path::config_file().as_path())
            .expect("Failed to read config file")
    }
}

#[derive(Deserialize)]
pub struct Repository {
    pub git_url: String,
}

impl Repository {
    pub fn dir(&self) -> PathBuf {
        repositories_dir().join(filenamify(&self.git_url))
    }
}
