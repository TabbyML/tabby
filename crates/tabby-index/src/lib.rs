//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.

mod code;
mod indexer;
mod tantivy_utils;

use indexer::{IndexAttributeBuilder, Indexer};
use tabby_common::index::corpus;

mod doc;

pub mod public {
    use super::*;
    pub use super::{
        code::CodeIndexer,
        doc::public::{DocIndexer, WebDocument},
    };

    pub fn run_index_garbage_collection(active_sources: Vec<String>) -> anyhow::Result<()> {
        for corpus in corpus::ALL.iter() {
            let indexer = Indexer::new(corpus);
            indexer.garbage_collect(&active_sources)?;
            indexer.commit();
        }

        Ok(())
    }
}
