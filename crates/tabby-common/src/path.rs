use std::{cell::Cell, env, path::PathBuf, sync::Mutex};

use lazy_static::lazy_static;

lazy_static! {
    static ref TABBY_ROOT: Mutex<Cell<PathBuf>> = {
        Mutex::new(Cell::new(match env::var("TABBY_ROOT") {
            Ok(x) => PathBuf::from(x),
            Err(_) => PathBuf::from(env::var("HOME").unwrap()).join(".tabby"),
        }))
    };
}

#[cfg(feature = "testutils")]
pub fn set_tabby_root(path: PathBuf) {
    println!("SET TABBY ROOT: '{}'", path.display());
    let cell = TABBY_ROOT.lock().unwrap();
    cell.replace(path);
}

fn tabby_root() -> PathBuf {
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

pub fn dataset_dir() -> PathBuf {
    tabby_root().join("dataset")
}

pub fn models_dir() -> PathBuf {
    tabby_root().join("models")
}

pub fn events_dir() -> PathBuf {
    tabby_root().join("events")
}

pub struct ModelDir(PathBuf);

impl ModelDir {
    pub fn new(model: &str) -> Self {
        Self(models_dir().join(model))
    }

    pub fn from(path: &str) -> Self {
        Self(PathBuf::from(path))
    }

    pub fn path(&self) -> &PathBuf {
        &self.0
    }

    pub fn path_string(&self, name: &str) -> String {
        self.0.join(name).display().to_string()
    }

    pub fn cache_info_file(&self) -> String {
        self.path_string(".cache_info.json")
    }

    pub fn metadata_file(&self) -> String {
        self.path_string("tabby.json")
    }

    pub fn tokenizer_file(&self) -> String {
        self.path_string("tokenizer.json")
    }

    pub fn ctranslate2_dir(&self) -> String {
        self.path_string("ctranslate2")
    }

    pub fn ggml_model_file(&self) -> String {
        self.path_string("ggml/default.gguf")
    }
}
