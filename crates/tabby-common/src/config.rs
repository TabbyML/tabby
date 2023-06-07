use std::{path::PathBuf, io::{ErrorKind, Error}};

use filenamify::filenamify;
use serde::Deserialize;

use crate::path::{repositories_dir, config_file};

#[derive(Deserialize)]
pub struct Config {
    pub repositories: Vec<Repository>,
}

impl Config {
    pub fn load() -> Result<Self,Error> {
        let file = serdeconv::from_toml_file(crate::path::config_file().as_path());
        match file {
            Ok(file) => Ok(file),
            Err(_) => Err(std::io::Error::new(ErrorKind::InvalidData, format!("Config {:?} doesn't exist or is not valid", config_file())))
        }
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
