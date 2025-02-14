use std::sync::Arc;

use async_stream::stream;
use futures::StreamExt;
use tabby_common::index::corpus;
use tabby_inference::Embedding;

use super::types::CommitHistory;
use crate::{
    commit::CommitHistoryBuilder,
    indexer::{Indexer, TantivyDocBuilder},
};

pub struct CommitHistoryIndexer {
    builder: TantivyDocBuilder<CommitHistory>,
    indexer: Indexer,
}

impl CommitHistoryIndexer {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        let builder = CommitHistoryBuilder::new(embedding);
        let builder = TantivyDocBuilder::new(corpus::STRUCTURED_DOC, builder);
        let indexer = Indexer::new(corpus::COMMIT_HISTORY);
        Self { indexer, builder }
    }

    pub async fn sync(&self, document: CommitHistory) -> bool {
        if !self.require_updates(&document) {
            return false;
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

    fn require_updates(&self, document: &CommitHistory) -> bool {
        true
    }
}
