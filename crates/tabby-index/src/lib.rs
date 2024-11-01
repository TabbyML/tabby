//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.

mod code;
mod indexer;
mod tantivy_utils;

use indexer::{IndexAttributeBuilder, Indexer};

mod doc;
mod structured_doc;

pub mod public {
    use indexer::IndexGarbageCollector;

    use super::*;
    pub use super::{
        code::CodeIndexer,
        doc::public::{DocIndexer, WebDocument},
        structured_doc::public::{StructuredDoc, StructuredDocIndexer},
    };

    pub fn run_index_garbage_collection(active_sources: Vec<String>) -> anyhow::Result<()> {
        let index_garbage_collector = IndexGarbageCollector::new();
        index_garbage_collector.garbage_collect(&active_sources)?;
        index_garbage_collector.commit();
        Ok(())
    }
}
