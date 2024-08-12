//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing repositories and updating indices.

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

    pub fn run_index_garbage_collection(
        active_sources: Vec<(String, String)>,
    ) -> anyhow::Result<()> {
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
}
