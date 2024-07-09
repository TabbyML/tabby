//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.

mod code;
mod indexer;


pub use doc::public::{DocIndexer, WebDocument};
use futures::StreamExt;
use indexer::{IndexAttributeBuilder, Indexer};
use tabby_common::index::corpus;


mod doc;


pub mod public {
    pub use super::{
        code::CodeIndexer,
        doc::public::{DocIndexer, WebDocument},
    };
}

pub fn run_index_garbage_collection(active_sources: Vec<(String, String)>) -> anyhow::Result<()> {
    let corpus_list = [corpus::WEB, corpus::CODE];
    for corpus in corpus_list.iter() {
        let active_sources: Vec<_> = active_sources
            .iter()
            .filter(|(c, _)| c == corpus)
            .map(|(_, source_id)| source_id.to_owned())
            .collect();
        let indexer = Indexer::new(corpus);
        indexer.garbage_collect(&active_sources)?;
        indexer.commit();
    }

    Ok(())
}

mod tantivy_utils {
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
}
