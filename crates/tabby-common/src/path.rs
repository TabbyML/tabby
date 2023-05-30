use std::env;
use std::path::PathBuf;

fn get_root_dir() -> PathBuf {
    match env::var("TABBY_ROOT") {
        Ok(x) => PathBuf::from(x),
        Err(_) => PathBuf::from(env::var("HOME").unwrap()).join(".tabby"),
    }
}

pub struct ModelDir(PathBuf);

impl ModelDir {
    pub fn new(model: &str) -> Self {
        Self(get_root_dir().join("models").join(model))
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

    pub fn metadata_file(&self) -> String {
        return self.path_string("metadata.json");
    }

    pub fn tokenizer_file(&self) -> String {
        return self.path_string("tokenizer.json");
    }

    pub fn ctranslate2_dir(&self) -> String {
        self.path_string("ctranslate2")
    }
}
