use std::path::PathBuf;

use tabby_common::path::tabby_root;

fn tabby_ee_root() -> PathBuf {
    tabby_root().join("ee")
}

pub fn db_file() -> PathBuf {
    tabby_ee_root().join("db.sqlite")
}
