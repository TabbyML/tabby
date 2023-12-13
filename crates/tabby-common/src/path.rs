use std::{cell::Cell, env, path::PathBuf, sync::Mutex};

use lazy_static::lazy_static;

lazy_static! {
    static ref TABBY_ROOT: Mutex<Cell<PathBuf>> = {
        Mutex::new(Cell::new(match env::var("TABBY_ROOT") {
            Ok(x) => PathBuf::from(x),
            Err(_) => home::home_dir().unwrap().join(".tabby"),
        }))
    };
}

#[cfg(feature = "testutils")]
pub fn set_tabby_root(path: PathBuf) {
    println!("SET TABBY ROOT: '{}'", path.display());
    let cell = TABBY_ROOT.lock().unwrap();
    cell.replace(path);
}

pub fn tabby_root() -> PathBuf {
    let mut cell = TABBY_ROOT.lock().unwrap();
    cell.get_mut().clone()
}

pub fn config_file() -> PathBuf {
    tabby_root().join("config.toml")
}

pub fn usage_id_file() -> PathBuf {
    tabby_root().join("usage_anonymous_id")
}

pub fn repositories_dir() -> PathBuf {
    tabby_root().join("repositories")
}

pub fn dependency_file() -> PathBuf {
    dataset_dir().join("deps.json")
}

pub fn index_dir() -> PathBuf {
    tabby_root().join("index")
}

pub fn dataset_dir() -> PathBuf {
    tabby_root().join("dataset")
}

pub fn models_dir() -> PathBuf {
    tabby_root().join("models")
}

pub fn events_dir() -> PathBuf {
    tabby_root().join("events")
}

mod registry {}
