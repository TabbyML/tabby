use std::{
    io::{Error, ErrorKind},
    path::PathBuf,
};

use filenamify::filenamify;
use serde::Deserialize;

use crate::path::{config_file, repositories_dir};

#[derive(Deserialize, Default)]
pub struct Config {
    pub repositories: Vec<Repository>,
    pub experimental: Experimental,
}

#[derive(Deserialize, Default)]
pub struct Experimental {
    pub enable_prompt_rewrite: bool,
}

impl Config {
    pub fn load() -> Result<Self, Error> {
        let file = serdeconv::from_toml_file(crate::path::config_file().as_path());
        file.map_err(|_| {
            Error::new(
                ErrorKind::InvalidData,
                format!("Config {:?} doesn't exist or is not valid", config_file()),
            )
        })
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
