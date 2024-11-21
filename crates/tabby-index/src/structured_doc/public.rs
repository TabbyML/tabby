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

/// StructuredDocState is used to track the state of the document source.
/// It is used to determine whether the document should be updated or deleted.
pub struct StructuredDocState {
    // updated_at is the time when the document was last updated.
    pub updated_at: DateTime<Utc>,
    // deleted indecates whether the document should be deleted in indexer
    // for example, a closed pull request will be marked as deleted, and
    // the indexer will remove it from the index.
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
