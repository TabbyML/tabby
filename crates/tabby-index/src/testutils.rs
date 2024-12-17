use std::path::PathBuf;

use tabby_common::config::{config_index_to_id, CodeRepository};

pub fn get_tabby_root() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("testdata");
    path
}

pub fn get_repository_config() -> CodeRepository {
    CodeRepository::new("https://github.com/TabbyML/tabby", &config_index_to_id(0))
}

pub fn get_rust_source_file() -> PathBuf {
    let mut path = get_tabby_root();
    path.push("repositories");
    path.push("https_github.com_TabbyML_tabby");
    path.push("rust.rs");
    path
}
