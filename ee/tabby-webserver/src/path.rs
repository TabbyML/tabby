use std::path::PathBuf;

use tabby_common::path::tabby_root;

pub fn tabby_ee_root() -> PathBuf {
    tabby_root().join("ee")
}

pub fn db_file() -> PathBuf {
    if cfg!(feature = "prod") {
        tabby_ee_root().join("db.sqlite")
    } else {
        tabby_ee_root().join("dev-db.sqlite")
    }
}

pub fn background_jobs_dir() -> PathBuf {
    if cfg!(feature = "prod") {
        tabby_ee_root().join("jobs")
    } else {
        tabby_ee_root().join("dev-jobs")
    }
}
