use anyhow::Result;
use async_stream::stream;
use chrono::{DateTime, TimeZone, Utc};
use futures::stream::{BoxStream, StreamExt};
use git2::{Repository, Sort};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone)]
pub struct Commit {
    pub id: String,
    pub message: String,
    pub author_name: String,
    pub author_email: String,
    pub author_at: DateTime<Utc>,
    pub committer_name: String,
    pub committer_email: String,
    pub commit_at: DateTime<Utc>,
}

impl From<git2::Commit<'_>> for Commit {
    fn from(commit: git2::Commit) -> Self {
        let author = commit.author();
        let committer = commit.committer();

        Self {
            id: commit.id().to_string(),
            message: commit.message().unwrap_or("").to_string(),
            author_name: author.name().unwrap_or("").to_string(),
            author_email: author.email().unwrap_or("").to_string(),
            author_at: Utc
                .timestamp_opt(author.when().seconds(), 0)
                .single()
                .unwrap_or_default(),
            committer_name: committer.name().unwrap_or("").to_string(),
            committer_email: committer.email().unwrap_or("").to_string(),
            commit_at: Utc
                .timestamp_opt(committer.when().seconds(), 0)
                .single()
                .unwrap_or_default(),
        }
    }
}

pub fn stream_commits(
    repo_path: String,
) -> (BoxStream<'static, Result<Commit>>, oneshot::Sender<()>) {
    let (stop_tx, stop_rx) = oneshot::channel();
    let (tx, mut rx) = mpsc::channel(16);

    // Spawn git operations in a tokio task
    tokio::spawn({
        let mut stop_rx = stop_rx;
        let tx_data = tx.clone();
        async move {
            // Keep all git operations inside spawn_blocking

            let result = tokio::task::spawn_blocking(move || {
                let repo = match Repository::open(&repo_path) {
                    Ok(repo) => repo,
                    Err(e) => return Err(anyhow::anyhow!("Failed to open repository: {}", e)),
                };

                let mut revwalk = match repo.revwalk() {
                    Ok(walk) => walk,
                    Err(e) => return Err(anyhow::anyhow!("Failed to create revwalk: {}", e)),
                };

                revwalk.push_head().ok();
                revwalk.set_sorting(Sort::TIME).ok();

                // Process commits inside the blocking task
                for oid in revwalk {
                    if stop_rx.try_recv().is_ok() {
                        break;
                    }

                    match oid.and_then(|oid| repo.find_commit(oid)) {
                        Ok(commit) => {
                            let commit: Commit = commit.into();
                            if tx_data.blocking_send(Ok(commit)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            if tx_data
                                .blocking_send(Err(anyhow::anyhow!("Failed to get commit: {}", e)))
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                }
                Ok(())
            })
            .await;

            if let Err(e) = result {
                tx.send(Err(anyhow::anyhow!("Task failed: {}", e)))
                    .await
                    .ok();
            }
        }
    });

    let s = stream! {
        while let Some(result) = rx.recv().await {
            yield result;
        }
    }
    .boxed();

    (s, stop_tx)
}
