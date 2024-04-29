use std::path::PathBuf;

use tabby_common::path::tabby_root;

pub fn incremental_repository_store() -> PathBuf {
    tabby_root().join("incremental-repositorries.kv")
}
