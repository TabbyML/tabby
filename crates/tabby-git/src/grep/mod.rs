mod output;
mod query;
mod searcher;

use std::path::PathBuf;

use anyhow::Context;
use async_stream::stream;
use futures::Stream;
use git2::TreeWalkResult;
pub use query::GrepQuery;
use searcher::GrepSearcher;
use tracing::{debug, warn};

use super::{bytes2path, rev_to_commit};

pub struct GrepFile {
    pub path: PathBuf,
    pub lines: Vec<GrepLine>,
}

pub struct GrepLine {
    /// Content of the line.
    pub line: GrepTextOrBase64,

    /// Byte offset in the file to the start of the line.
    pub byte_offset: usize,

    /// Line number in the file, starting from 1.
    pub line_number: usize,

    /// The matches in the line.
    pub sub_matches: Vec<GrepSubMatch>,
}

pub enum GrepTextOrBase64 {
    Text(String),
    Base64(Vec<u8>),
}

pub struct GrepSubMatch {
    // Byte offsets in the line
    pub bytes_start: usize,
    pub bytes_end: usize,
}

pub fn grep(
    repository: git2::Repository,
    rev: Option<&str>,
    query: &GrepQuery,
) -> anyhow::Result<impl Stream<Item = GrepFile>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);

    let rev = rev.map(|s| s.to_owned());
    let query = query.clone();
    debug!("{:?}", query);
    let searcher = query.searcher()?;
    let task =
        tokio::task::spawn_blocking(move || grep_impl(repository, rev.as_deref(), searcher, tx));

    Ok(stream! {
        while let Some(file) = rx.recv().await {
            yield file;
        }

        if let Err(err) = task.await {
            warn!("Error grepping repository: {}", err);
        }
    })
}

fn grep_impl(
    repository: git2::Repository,
    rev: Option<&str>,
    mut searcher: GrepSearcher,
    tx: tokio::sync::mpsc::Sender<GrepFile>,
) -> anyhow::Result<()> {
    let commit = rev_to_commit(&repository, rev)?;
    let tree = commit.tree()?;

    tree.walk(git2::TreeWalkMode::PreOrder, |path, entry| {
        // Skip non-blob entries
        if entry.kind() != Some(git2::ObjectType::Blob) {
            return TreeWalkResult::Ok;
        }

        match grep_file(&repository, &mut searcher, path, entry, tx.clone()) {
            Ok(()) => {}
            Err(e) => {
                warn!("Error grepping file: {}", e);
            }
        }
        TreeWalkResult::Ok
    })?;
    Ok(())
}

fn grep_file(
    repository: &git2::Repository,
    searcher: &mut GrepSearcher,
    path: &str,
    entry: &git2::TreeEntry,
    tx: tokio::sync::mpsc::Sender<GrepFile>,
) -> anyhow::Result<()> {
    let object = entry.to_object(repository)?;
    let content = object.as_blob().context("Not a blob")?.content();

    let path = PathBuf::from(path).join(bytes2path(entry.name_bytes()));

    let mut output = output::GrepOutput::new(path.clone(), tx.clone());
    searcher.search(content, &mut output)?;
    output.flush(
        searcher.require_file_match,
        searcher.require_content_match,
        content,
    );

    Ok(())
}
#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::testutils::TempGitRepository;

    #[tokio::test]
    async fn test_grep() {
        let root = TempGitRepository::default();
        let query = GrepQuery::builder().pattern("crosscodeeval_data").build();

        let files: Vec<_> = grep(root.repository(), None, &query)
            .unwrap()
            .collect()
            .await;
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, PathBuf::from("203_llm_evaluation/README.md"));

        let query = GrepQuery::builder().pattern("ideas").build();
        let files: Vec<_> = grep(root.repository(), None, &query)
            .unwrap()
            .collect()
            .await;
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, PathBuf::from("README.md"));

        let query = GrepQuery::builder()
            .file_type("markdown")
            .file_pattern("llm_evaluation")
            .build();
        let files: Vec<_> = grep(root.repository(), None, &query)
            .unwrap()
            .collect()
            .await;
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, PathBuf::from("203_llm_evaluation/README.md"));

        // File patterns are AND-ed.
        let query = GrepQuery::builder()
            .file_pattern(".md")
            .file_pattern("llm_evaluation")
            .build();
        let files: Vec<_> = grep(root.repository(), None, &query)
            .unwrap()
            .collect()
            .await;
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, PathBuf::from("203_llm_evaluation/README.md"));

        // When positive condition provided, return nothing if no matches.
        let query = GrepQuery::builder()
            .file_type("markdown")
            .pattern("non_exist_pattern")
            .build();
        let files: Vec<_> = grep(root.repository(), None, &query)
            .unwrap()
            .collect()
            .await;
        assert_eq!(files.len(), 0);

        // When no positive condition provided, all
        // files not matching negative conditions should be returned.
        let query = GrepQuery::builder()
            .negative_file_type("rust")
            .negative_pattern("non_exist_pattern")
            .build();
        let files: Vec<_> = grep(root.repository(), None, &query)
            .unwrap()
            .collect()
            .await;
        assert_eq!(files.len(), 9);

        let query = GrepQuery::builder()
            .pattern("non_exist_pattern")
            .negative_file_pattern("non_exist_pattern")
            .negative_pattern("ideas")
            .build();
        let files: Vec<_> = grep(root.repository(), None, &query)
            .unwrap()
            .collect()
            .await;
        assert_eq!(files.len(), 0);

        let query = GrepQuery::builder().build();
        assert!(grep(root.repository(), None, &query).is_err());
    }
}
