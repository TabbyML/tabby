use std::{pin::pin, sync::Arc};

use futures::StreamExt;
use tabby_common::{config::CodeRepository, index::structured_doc::fields::commit};
use tabby_git::stream_commits;
use tabby_index::public::{
    StructuredDoc, StructuredDocCommitFields, StructuredDocFields, StructuredDocIndexer,
    STRUCTURED_DOC_KIND_COMMIT,
};
use tabby_inference::Embedding;
use tabby_schema::Result;

const MAX_COMMIT_HISTORY_COUNT: usize = 10000;

fn to_commit_document(commit: &tabby_git::Commit) -> StructuredDocCommitFields {
    StructuredDocCommitFields {
        sha: commit.id.clone(),
        message: commit.message.clone(),
        author_email: commit.author_email.clone(),
        author_at: commit.author_at,
    }
}

// Do not reuse the indexer for indexing and garbage collection.
// The indexer must be committed after indexing,
// or the garbage collection is unable to read the latest index.
pub async fn refresh(embedding: Arc<dyn Embedding>, repository: &CodeRepository) -> Result<()> {
    logkit::info!("Building commit index: {}", repository.canonical_git_url());
    indexing(embedding.clone(), repository).await;

    logkit::info!(
        "Garbage collecting commit index: {}",
        repository.canonical_git_url()
    );
    garbage_collection(embedding.clone(), &repository.source_id).await;

    Ok(())
}

struct StopGuard(Option<tokio::sync::oneshot::Sender<()>>);
impl Drop for StopGuard {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            sender.send(()).ok();
        }
    }
}

async fn indexing(embedding: Arc<dyn Embedding>, repository: &CodeRepository) {
    let indexer = StructuredDocIndexer::new(embedding);

    let (commits, stop_tx) = stream_commits(repository.dir().to_string_lossy().to_string());
    let mut commits = pin!(commits);
    // Will automatically send stop signal when it goes out of scope
    let _stop_guard = StopGuard(Some(stop_tx));

    let mut count = 0;
    let mut num_updated = 0;

    while let Some(commit_result) = commits.next().await {
        if count >= MAX_COMMIT_HISTORY_COUNT {
            break;
        }

        match commit_result {
            Ok(commit) => {
                count += 1;
                let commit = StructuredDoc {
                    source_id: repository.source_id.clone(),
                    fields: StructuredDocFields::Commit(to_commit_document(&commit)),
                };
                if !indexer.sync(commit).await {
                    continue;
                }

                num_updated += 1;
                if count % 100 == 0 {
                    logkit::info!("{} commits seen, {} commits updated", count, num_updated);
                }
            }
            Err(e) => {
                logkit::warn!("Failed to process commit: {}", e);
                continue;
            }
        }
    }

    logkit::info!("{} commits seen, {} commits updated", count, num_updated);
    indexer.commit();
}

async fn garbage_collection(embedding: Arc<dyn Embedding>, source_id: &str) {
    let indexer = StructuredDocIndexer::new(embedding);

    let count = match indexer
        .count_doc(source_id, STRUCTURED_DOC_KIND_COMMIT)
        .await
    {
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
        .list_latest_ids(
            source_id,
            STRUCTURED_DOC_KIND_COMMIT,
            commit::AUTHOR_AT,
            MAX_COMMIT_HISTORY_COUNT,
        )
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
        indexer.delete(&commit).await;
    }

    logkit::info!(
        "Finished garbage collection for commit history: {count} items, deleted {num_deleted}"
    );
    indexer.commit();
}
