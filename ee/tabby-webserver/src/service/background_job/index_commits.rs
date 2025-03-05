use std::{pin::pin, sync::Arc};

use futures::StreamExt;
use tabby_common::{config::CodeRepository, index::structured_doc::fields::commit};
use tabby_git::stream_commits;
use tabby_index::public::{
    StructuredDoc, StructuredDocCommitDiff, StructuredDocCommitFields, StructuredDocFields,
    StructuredDocIndexer, STRUCTURED_DOC_KIND_COMMIT,
};
use tabby_inference::Embedding;
use tabby_schema::Result;

const MAX_COMMIT_HISTORY_COUNT: usize = 100;

fn to_commit_document<'a>(
    commit: &tabby_git::Commit,
    repo: &CodeRepository,
) -> StructuredDocCommitFields {
    let diff = commit
        .diff
        .iter()
        .map(|diff| StructuredDocCommitDiff {
            path: diff.path.clone(),
            content: diff.content.clone(),
        })
        .collect();
    StructuredDocCommitFields {
        git_url: repo.canonical_git_url().clone(),
        sha: commit.id.clone(),
        message: commit.message.clone(),
        author_email: commit.author_email.clone(),
        author_at: commit.author_at,
        committer_email: commit.committer_email.clone(),
        commit_at: commit.commit_at,

        diff,
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
    garbage_collection(embedding.clone(), &repository.canonical_git_url()).await;

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

    let existing_commit_count = indexer
        .count_doc_by_attribute(
            STRUCTURED_DOC_KIND_COMMIT,
            commit::GIT_URL,
            &repository.source_id,
        )
        .await
        .unwrap_or(0);

    while let Some(commit_result) = commits.next().await {
        if existing_commit_count + count >= MAX_COMMIT_HISTORY_COUNT * 2 {
            break;
        }

        match commit_result {
            Ok(commit) => {
                let commit = StructuredDoc {
                    source_id: repository.source_id.clone(),
                    fields: StructuredDocFields::Commit(to_commit_document(&commit, repository)),
                };
                if !indexer.sync(commit).await {
                    break;
                }

                num_updated += 1;
                count += 1;
                if count % 50 == 0 {
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

async fn garbage_collection(embedding: Arc<dyn Embedding>, repo: &str) {
    let indexer = StructuredDocIndexer::new(embedding);

    let count = match indexer
        .count_doc_by_attribute(STRUCTURED_DOC_KIND_COMMIT, commit::GIT_URL, repo)
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
        .get_newest_ids_by_attribute(
            STRUCTURED_DOC_KIND_COMMIT,
            commit::GIT_URL,
            repo,
            count,
            100,
            commit::COMMIT_AT,
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
