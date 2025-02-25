use std::sync::Arc;

use async_stream::stream;
use futures::StreamExt;
use tabby_common::index::{commit::fields, corpus};
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

const MAX_COMMIT_HISTORY_COUNT: usize = 100;

impl CommitHistoryIndexer {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        let builder = CommitHistoryBuilder::new(embedding);
        let builder = TantivyDocBuilder::new(corpus::COMMIT_HISTORY, builder);
        let indexer = Indexer::new(corpus::COMMIT_HISTORY);
        Self { indexer, builder }
    }

    pub async fn sync(&self, commit: CommitHistory) -> bool {
        if !self.require_updates(&commit).await {
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

    async fn require_updates(&self, commit: &CommitHistory) -> bool {
        if self.indexer.is_indexed(commit.to_index_id().id.as_str()) {
            return false;
        }

        // Save up to 2 * MAX_COMMIT_HISTORY_COUNT items
        // older ones will be purged by garbage collection.
        //
        // This is to prevent the index from growing indefinitely or
        // exceeding the MAX_COMMIT_HISTORY_COUNT between two synchronizations.
        if self
            .indexer
            .count_doc_by_attribute(fields::GIT_URL, &commit.git_url)
            .await
            .unwrap_or(0)
            > MAX_COMMIT_HISTORY_COUNT * 2
        {
            return false;
        }

        true
    }
}

pub async fn garbage_collection(repo: &str) {
    let indexer = Indexer::new(corpus::COMMIT_HISTORY);
    let count = match indexer.count_doc_by_attribute(fields::GIT_URL, repo).await {
        Ok(count) => count,
        Err(err) => {
            logkit::warn!(
                "Failed to count commit history for garbage collection: {}",
                err
            );
            return;
        }
    };

    let old_commits = match indexer
        .get_newest_ids_by_attribute(fields::GIT_URL, repo, count, 100, fields::COMMIT_AT)
        .await
    {
        Ok(commits) => commits,
        Err(err) => {
            logkit::warn!(
                "Failed to list old commit history for garbage collection: {}",
                err
            );
            return;
        }
    };
    let num_deleted = old_commits.len();

    for commit in old_commits {
        indexer.delete(&commit);
    }

    logkit::info!(
        "Finished garbage collection for commit history: {count} items, deleted {num_deleted}"
    );
    indexer.commit();
}
