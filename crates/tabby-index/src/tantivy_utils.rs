use std::{fs, path::Path};

use tantivy::{directory::MmapDirectory, schema::Schema, Index};
use tracing::{debug, warn};

pub fn open_or_create_index(code: &Schema, path: &Path) -> (bool, Index) {
    let (recreated, index) = match open_or_create_index_impl(code, path) {
        Ok(index) => (false, index),
        Err(err) => {
            warn!(
                "Failed to open index repositories: {}, removing index directory '{}'...",
                err,
                path.display()
            );
            fs::remove_dir_all(path).expect("Failed to remove index directory");

            debug!("Reopening index repositories...");
            (
                true,
                open_or_create_index_impl(code, path).expect("Failed to open index"),
            )
        }
    };
    (recreated, index)
}

fn open_or_create_index_impl(code: &Schema, path: &Path) -> tantivy::Result<Index> {
    fs::create_dir_all(path).expect("Failed to create index directory");
    let directory = MmapDirectory::open(path).expect("Failed to open index directory");
    Index::open_or_create(directory, code.clone())
}
