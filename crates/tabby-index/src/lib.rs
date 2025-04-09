//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.

mod code;
mod indexer;
mod tantivy_utils;

#[cfg(test)]
mod testutils;

use indexer::{IndexAttributeBuilder, Indexer};

mod structured_doc;

#[cfg(test)]
mod indexer_tests;

pub mod public {
    use indexer::IndexGarbageCollector;

    use super::*;
    pub use super::{
        code::CodeIndexer,
        structured_doc::public::{
            StructuredDoc, StructuredDocCommitFields, StructuredDocFields, StructuredDocIndexer,
            StructuredDocIssueFields, StructuredDocPageFields, StructuredDocPullDocumentFields,
            StructuredDocState, StructuredDocWebFields, KIND_COMMIT as STRUCTURED_DOC_KIND_COMMIT,
            KIND_ISSUE as STRUCTURED_DOC_KIND_ISSUE, KIND_PAGE as STRUCTURED_DOC_KIND_PAGE,
            KIND_PULL as STRUCTURED_DOC_KIND_PULL, KIND_WEB as STRUCTURED_DOC_KIND_WEB,
        },
    };

    pub fn run_index_garbage_collection(active_sources: Vec<String>) -> anyhow::Result<()> {
        let index_garbage_collector = IndexGarbageCollector::new();
        index_garbage_collector.garbage_collect(&active_sources)?;
        index_garbage_collector.commit();
        Ok(())
    }
}
