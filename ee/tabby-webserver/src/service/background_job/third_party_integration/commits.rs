use std::{pin::pin, sync::Arc};

use futures::StreamExt;
use lettre::transport::smtp::response::Code;
use tabby_common::config::CodeRepository;
use tabby_git::stream_commits;
use tabby_index::public::{CommitHistory, CommitHistoryIndexer};
use tabby_inference::Embedding;
use tabby_schema::Result;
use tracing::warn;

fn to_commit_history<'a>(commit: &tabby_git::Commit, repo: &CodeRepository) -> CommitHistory {
    CommitHistory {
        source_id: repo.source_id.clone(),

        git_url: repo.git_url.clone(),
        sha: commit.id.clone(),
        message: commit.message.clone(),
        author_email: commit.author_email.clone(),
        author_at: commit.author_at,
        committer_email: commit.committer_email.clone(),
        commit_at: commit.commit_at,

        diff: None,
    }
}

pub async fn refresh(embedding: Arc<dyn Embedding>, repository: &CodeRepository) -> Result<()> {
    logkit::info!(
        "Building commit history index: {}",
        repository.canonical_git_url()
    );

    let indexer = CommitHistoryIndexer::new(embedding);

    indexing(indexer, repository).await;

    // clear the index older/more than 100 commits
    // indexer.garbage_collection().await;

    Ok(())
}

pub async fn indexing(indexer: CommitHistoryIndexer, repository: &CodeRepository) {
    let (commits, stop_tx) = stream_commits(repository.dir().to_string_lossy().to_string());
    let mut commits = pin!(commits);

    let mut count = 0;
    let mut num_updated = 0;

    while let Some(commit_result) = commits.next().await {
        match commit_result {
            Ok(commit) => {
                let commit = to_commit_history(&commit, repository);
                if !indexer.sync(commit).await {
                    // We synchronize commits based on their date stamps,
                    // sync returns false if the commit has already been indexed, indicating that no update is necessary.
                    // Halt the stream and break the loop
                    stop_tx.send(()).ok();
                    break;
                }
                count += 1;
                if count % 100 == 0 {
                    logkit::info!("{} commits seen, {} commits updated", count, num_updated);
                }
                num_updated += 1;
            }
            Err(e) => {
                warn!("Failed to process commit: {}", e);
                continue;
            }
        }
    }

    logkit::info!("{} commits seen, {} commits updated", count, num_updated);
    indexer.commit();
}
