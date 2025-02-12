use std::path::PathBuf;

use async_stream::stream;
use futures::{Stream, StreamExt};
use git2::TreeWalkResult;

use super::rev_to_commit;
use crate::bytes2path;

pub struct GitFileSearch {
    pub r#type: &'static str,
    pub path: String,

    /// matched indices for fuzzy search query.
    pub indices: Vec<i32>,
}

impl GitFileSearch {
    fn new(r#type: &'static str, path: String, indices: Vec<u32>) -> Self {
        Self {
            r#type,
            path,
            indices: indices.into_iter().map(|i| i as i32).collect(),
        }
    }
}

fn walk(
    repository: git2::Repository,
    rev: Option<&str>,
    tx: tokio::sync::mpsc::Sender<(bool, PathBuf)>,
) -> anyhow::Result<()> {
    let commit = rev_to_commit(&repository, rev)?;
    let tree = commit.tree()?;

    tree.walk(git2::TreeWalkMode::PreOrder, |path, entry| {
        let is_file = entry.kind() == Some(git2::ObjectType::Blob);
        let path = PathBuf::from(path).join(bytes2path(entry.name_bytes()));
        match tx.blocking_send((is_file, path)) {
            Ok(_) => TreeWalkResult::Ok,
            Err(_) => TreeWalkResult::Abort,
        }
    })?;

    Ok(())
}

fn walk_bfs(
    repository: git2::Repository,
    rev: Option<&str>,
    tx: tokio::sync::mpsc::Sender<(bool, PathBuf)>,
) -> anyhow::Result<()> {
    use std::collections::VecDeque;

    let commit = rev_to_commit(&repository, rev)?;
    let tree = commit.tree()?;

    // Initialize queue with root tree
    let mut queue = VecDeque::new();
    queue.push_back(("".to_string(), tree));

    while let Some((current_path, current_tree)) = queue.pop_front() {
        // Process all entries in current directory
        for entry in current_tree.iter() {
            let entry_name = bytes2path(entry.name_bytes());
            let path = if current_path.is_empty() {
                PathBuf::from(&entry_name)
            } else {
                PathBuf::from(&current_path).join(entry_name)
            };

            let is_file = entry.kind() == Some(git2::ObjectType::Blob);

            if let Err(_e) = tx.blocking_send((is_file, path.clone())) {
                return Err(anyhow::anyhow!("Failed to send path to channel"));
            }

            // If it's a directory, add to queue for later processing
            match entry.to_object(&repository) {
                Ok(obj) => {
                    if let Ok(subtree) = obj.peel_to_tree() {
                        queue.push_back((path.display().to_string(), subtree));
                        eprintln!("Added directory to queue: {}", path.display());
                    }
                }
                Err(e) => {
                    eprintln!("Error processing entry {:?}: {}", path, e);
                }
            }
        }
    }

    Ok(())
}

async fn walk_stream(
    repository: git2::Repository,
    rev: Option<&str>,
) -> impl Stream<Item = (bool, PathBuf)> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);

    let rev = rev.map(|s| s.to_owned());
    let task = tokio::task::spawn_blocking(move || walk(repository, rev.as_deref(), tx));

    stream! {
        while let Some(value) = rx.recv().await {
            yield value;
        }

        let _ = task.await;
    }
}

async fn walk_stream_bfs(
    repository: git2::Repository,
    rev: Option<&str>,
) -> impl Stream<Item = (bool, PathBuf)> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);

    let rev = rev.map(|s| s.to_owned());
    let task = tokio::task::spawn_blocking(move || walk_bfs(repository, rev.as_deref(), tx));

    stream! {
        while let Some(value) = rx.recv().await {
            yield value;
        }

        let _ = task.await;
    }
}

pub async fn search(
    repository: git2::Repository,
    rev: Option<&str>,
    pattern: &str,
    limit: usize,
) -> anyhow::Result<Vec<GitFileSearch>> {
    let mut scored_entries: Vec<(_, _)> = stream! {
        let mut nucleo = nucleo::Matcher::new(nucleo::Config::DEFAULT.match_paths());
        let needle = nucleo::pattern::Pattern::new(
            pattern,
            nucleo::pattern::CaseMatching::Ignore,
            nucleo::pattern::Normalization::Smart,
            nucleo::pattern::AtomKind::Fuzzy,
        );

        for await (is_file, basepath) in walk_stream(repository, rev).await {
            let r#type = if is_file { "file" } else { "dir" };
            let basepath = basepath.display().to_string();
            let haystack: nucleo::Utf32String = basepath.clone().into();
            let mut indices = Vec::new();
            let score = needle.indices(haystack.slice(..), &mut nucleo, &mut indices);
            if let Some(score) = score {
                yield (score, GitFileSearch::new(r#type, basepath, indices));
            }
        }
    }
    // Ensure there's at least 1000 entries with scores > 0 for quality.
    .take(1000)
    .collect()
    .await;

    scored_entries.sort_by_key(|x| -(x.0 as i32));
    let entries = scored_entries
        .into_iter()
        .map(|x| x.1)
        .take(limit)
        .collect();

    Ok(entries)
}

pub async fn list(
    repository: git2::Repository,
    rev: Option<&str>,
    limit: Option<usize>,
) -> anyhow::Result<Vec<GitFileSearch>> {
    let entries: Vec<GitFileSearch> = stream! {
        for await (is_file, basepath) in walk_stream(repository, rev).await {
            let r#type = if is_file { "file" } else { "dir" };
            let basepath = basepath.display().to_string();
            yield GitFileSearch::new(r#type, basepath, Vec::new());
        }
    }
    .take(limit.unwrap_or(usize::MAX))
    .collect()
    .await;

    Ok(entries)
}

pub async fn list_files_bfs(
    repository: git2::Repository,
    rev: Option<&str>,
    limit: Option<usize>,
) -> anyhow::Result<Vec<GitFileSearch>> {
    let entries: Vec<GitFileSearch> = stream! {
        for await (is_file, basepath) in walk_stream_bfs(repository, rev).await {
            let r#type = if is_file { "file" } else { "dir" };
            let basepath = basepath.display().to_string();
            yield GitFileSearch::new(r#type, basepath, Vec::new());
        }
    }
    .take(limit.unwrap_or(usize::MAX))
    .collect()
    .await;

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutils::TempGitRepository;

    #[tokio::test]
    async fn it_search() {
        let root = TempGitRepository::default();

        let result = search(root.repository(), None, "moonscript_lora md", 5)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].r#type, "file");
        assert_eq!(result[0].path, "201_lm_moonscript_lora/README.md");
    }
}
