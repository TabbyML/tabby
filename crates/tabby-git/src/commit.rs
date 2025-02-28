use std::cell::RefCell;

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

    pub diff: Vec<Diff>,
}

#[derive(Debug, Clone)]
pub struct Diff {
    pub path: String,
    pub content: String,
}

fn commit_from_git2(repo: &git2::Repository, commit: &git2::Commit) -> Commit {
    let author = commit.author();
    let committer = commit.committer();

    Commit {
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

        diff: get_diff_of_commit(repo, commit).unwrap_or_default(),
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
                            let commit: Commit = commit_from_git2(&repo, &commit);
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

fn get_diff_of_commit(repo: &git2::Repository, commit: &git2::Commit) -> Result<Vec<Diff>> {
    let tree = commit.tree()?;
    let parent_tree = if commit.parent_count() > 0 {
        commit.parent(0)?.tree()?
    } else {
        return Ok(vec![]);
    };

    let diff = repo.diff_tree_to_tree(
        Some(&parent_tree),
        Some(&tree),
        Some(
            git2::DiffOptions::new()
                .ignore_whitespace(true)
                .ignore_whitespace_change(true)
                .ignore_whitespace_eol(true)
                .ignore_submodules(true)
                .context_lines(0),
        ),
    )?;

    let result = RefCell::new(Vec::new());
    diff.foreach(
        &mut |delta, _| {
            if let Some(path) = delta.new_file().path() {
                if let Some(path_str) = path.to_str() {
                    result.borrow_mut().push(Diff {
                        path: path_str.to_string(),
                        content: String::new(),
                    });
                }
            }
            true
        },
        None,
        None,
        Some(&mut |_delta, _hunk, line| {
            if let Some(last) = result.borrow_mut().last_mut() {
                let prefix = match line.origin() {
                    '+' => "+",
                    '-' => "-",
                    _ => " ",
                };
                last.content.push_str(prefix);
                last.content
                    .push_str(&String::from_utf8_lossy(line.content()));
            }
            true
        }),
    )
    .unwrap_or_default();

    Ok(result.into_inner())
}
