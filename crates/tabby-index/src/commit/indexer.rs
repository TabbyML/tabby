use std::sync::Arc;

use async_stream::stream;
use futures::StreamExt;
use tabby_common::index::corpus;
use tabby_inference::Embedding;

use super::types::CommitHistory;
use crate::{
    commit::CommitHistoryBuilder,
    indexer::{Indexer, TantivyDocBuilder, ToIndexId},
};

pub struct CommitHistoryIndexer {
    builder: TantivyDocBuilder<CommitHistory>,
    indexer: Indexer,
}

impl CommitHistoryIndexer {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        let builder = CommitHistoryBuilder::new(embedding);
        let builder = TantivyDocBuilder::new(corpus::COMMIT_HISTORY, builder);
        let indexer = Indexer::new(corpus::COMMIT_HISTORY);
        Self { indexer, builder }
    }

    pub async fn sync(&self, commit: CommitHistory) -> bool {
        if !self.require_updates(&commit) {
            return false;
        }

        stream! {
            let (id, s) = self.builder.build(commit).await;
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

    fn require_updates(&self, commit: &CommitHistory) -> bool {
        !self.indexer.is_indexed(commit.to_index_id().id.as_str())
    }
}
