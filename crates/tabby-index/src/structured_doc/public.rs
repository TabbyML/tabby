use std::sync::Arc;

use async_stream::stream;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use tabby_common::index::corpus;
use tabby_inference::Embedding;

pub use super::types::{
    issue::IssueDocument as StructuredDocIssueFields,
    pull::PullRequest as StructuredDocPullRequestFields,
    web::WebDocument as StructuredDocWebFields, StructuredDoc, StructuredDocFields,
};
use super::{create_structured_doc_builder, types::BuildStructuredDoc};
use crate::{indexer::TantivyDocBuilder, Indexer};

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

    pub async fn add(&self, updated_at: DateTime<Utc>, document: StructuredDoc) -> bool {
        if document.should_skip() {
            return false;
        }

        if self.indexer.is_indexed_after(document.id(), updated_at) {
            return false;
        };

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

    pub fn commit(self) {
        self.indexer.commit();
    }
}
