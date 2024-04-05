use std::path::PathBuf;

pub fn repository_meta_db() -> PathBuf {
    tabby_common::path::tabby_root().join("repositories.kv")
}
