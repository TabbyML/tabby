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

async fn indexing(embedding: Arc<dyn Embedding>, repository: &CodeRepository) {
    let indexer = StructuredDocIndexer::new(embedding);

    let (commits, stop_tx) = stream_commits(repository.dir().to_string_lossy().to_string());
    let mut commits = pin!(commits);

    let mut count = 0;
    let mut num_updated = 0;

    while let Some(commit_result) = commits.next().await {
        match commit_result {
            Ok(commit) => {
                let commit = StructuredDoc {
                    source_id: repository.source_id.clone(),
                    fields: StructuredDocFields::Commit(to_commit_document(&commit, repository)),
                };
                if !indexer.sync(commit).await {
                    // We synchronize commits based on their date stamps,
                    // sync returns false if the commit has already been indexed, indicating that no update is necessary.
                    // Halt the stream and break the loop
                    stop_tx.send(()).ok();
                    break;
                }
                count += 1;
                if count % 50 == 0 {
                    logkit::info!("{} commits seen, {} commits updated", count, num_updated);
                }
                num_updated += 1;
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
