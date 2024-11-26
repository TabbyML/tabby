use std::sync::Arc;

use async_stream::stream;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use tabby_common::index::corpus;
use tabby_inference::Embedding;

pub use super::types::{
    issue::IssueDocument as StructuredDocIssueFields,
    pull::PullDocument as StructuredDocPullDocumentFields,
    web::WebDocument as StructuredDocWebFields, StructuredDoc, StructuredDocFields,
};
use super::{create_structured_doc_builder, types::BuildStructuredDoc};
use crate::{indexer::TantivyDocBuilder, Indexer};

/// StructuredDocState tracks the state of the document source.
/// It helps determine whether the document should be updated or deleted.
pub struct StructuredDocState {
    // updated_at is the time when the document was last updated.
    // when the updated_at is earlier than the document's index time,
    // the update will be skipped.
    pub updated_at: DateTime<Utc>,
    // deleted indicates whether the document should be removed from the indexer.
    // For instance, a closed pull request will be marked as deleted,
    // prompting the indexer to remove it from the index.
    pub deleted: bool,
}

pub struct StructuredDocIndexer {
    builder: TantivyDocBuilder<StructuredDoc>,
    indexer: Indexer,
}

impl StructuredDocIndexer {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        let builder = create_structured_doc_builder(embedding);
        let indexer = Indexer::new(corpus::STRUCTURED_DOC);
        Self { indexer, builder }
    }

    // The sync process updates the document in the indexer incrementally.
    // It first determines whether the document requires an update.
    //
    // If an update is needed, it checks the deletion state of the document.
    // If the document is marked as deleted, it will be removed.
    // Next, the document is rebuilt, the original is deleted, and the newly indexed document is added.
    pub async fn sync(&self, state: StructuredDocState, document: StructuredDoc) -> bool {
        if !self.require_updates(state.updated_at, &document) {
            return false;
        }

        if state.deleted {
            return self.delete(document.id()).await;
        }

        stream! {
            let (id, s) = self.builder.build(document).await;
            self.indexer.delete(&id);

            for await doc in s.buffer_unordered(std::cmp::max(std::thread::available_parallelism().unwrap().get() * 2, 32)) {
                if let Ok(Some(doc)) = doc {
                    self.indexer.add(doc).await;
                }
            }
        }.count().await;
        true
    }

    pub async fn delete(&self, id: &str) -> bool {
        if self.indexer.is_indexed(id) {
            self.indexer.delete(id);
            true
        } else {
            false
        }
    }

    pub fn commit(self) {
        self.indexer.commit();
    }

    fn require_updates(&self, updated_at: DateTime<Utc>, document: &StructuredDoc) -> bool {
        if document.should_skip() {
            return false;
        }

        if self.indexer.is_indexed_after(document.id(), updated_at)
            && !self.indexer.has_failed_chunks(document.id())
        {
            return false;
        };

        true
    }
}
