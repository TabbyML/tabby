use std::{cell::Cell, env, path::PathBuf, sync::Mutex};

use lazy_static::lazy_static;

lazy_static! {
    static ref TABBY_ROOT: Mutex<Cell<PathBuf>> = {
        Mutex::new(Cell::new(match env::var("TABBY_ROOT") {
            Ok(x) => PathBuf::from(x),
            Err(_) => home::home_dir().unwrap().join(".tabby"),
        }))
    };
    static ref TABBY_MODEL_CACHE_ROOT: Option<PathBuf> =
        env::var("TABBY_MODEL_CACHE_ROOT").ok().map(PathBuf::from);
}

#[cfg(any(feature = "testutils", test))]
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

pub fn index_dir() -> PathBuf {
    tabby_root().join("index")
}

pub fn models_dir() -> PathBuf {
    if let Some(cache_root) = &*TABBY_MODEL_CACHE_ROOT {
        cache_root.clone()
    } else {
        tabby_root().join("models")
    }
}

pub fn events_dir() -> PathBuf {
    tabby_root().join("events")
}

// FIXME: migrate to /corpus/code/cache
pub fn cache_dir() -> PathBuf {
    tabby_root().join("cache")
}

mod registry {}
