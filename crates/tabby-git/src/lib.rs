mod file_search;
mod serve_git;

mod grep;
use std::path::Path;

use axum::{
    body::Body,
    http::{Response, StatusCode},
};
use file_search::GitFileSearch;
use futures::Stream;
pub use grep::{GrepFile, GrepLine, GrepSubMatch, GrepTextOrBase64};

pub async fn search_files(
    root: &Path,
    rev: Option<&str>,
    pattern: &str,
    limit: usize,
) -> anyhow::Result<Vec<GitFileSearch>> {
    file_search::search(git2::Repository::open(root)?, rev, pattern, limit).await
}

pub async fn grep(
    root: &Path,
    rev: Option<&str>,
    query: &str,
) -> anyhow::Result<impl Stream<Item = GrepFile>> {
    let repository = git2::Repository::open(root)?;
    let query: grep::GrepQuery = query.parse()?;
    grep::grep(repository, rev, &query)
}

pub fn serve_file(
    root: &Path,
    commit: Option<&str>,
    path: Option<&str>,
) -> std::result::Result<Response<Body>, StatusCode> {
    let repository = git2::Repository::open(root).map_err(|_| StatusCode::NOT_FOUND)?;
    serve_git::serve(&repository, commit, path)
}

pub fn list_refs(root: &Path) -> anyhow::Result<Vec<String>> {
    let repository = git2::Repository::open(root)?;
    let refs = repository.references()?;
    Ok(refs
        .filter_map(|r| r.ok())
        .map(|r| r.name().unwrap().to_string())
        // Filter out remote refs
        .filter(|r| !r.starts_with("refs/remotes/"))
        .collect())
}

fn rev_to_commit<'a>(
    repository: &'a git2::Repository,
    rev: Option<&str>,
) -> anyhow::Result<git2::Commit<'a>> {
    let commit = match rev {
        Some(rev) => repository.revparse_single(rev)?.peel_to_commit()?,
        None => repository.head()?.peel_to_commit()?,
    };
    Ok(commit)
}

#[cfg(unix)]
pub fn bytes2path(b: &[u8]) -> &Path {
    use std::os::unix::prelude::*;
    Path::new(std::ffi::OsStr::from_bytes(b))
}
#[cfg(windows)]
pub fn bytes2path(b: &[u8]) -> &Path {
    use std::str;
    Path::new(str::from_utf8(b).unwrap())
}

#[cfg(test)]
mod testutils {
    use std::process::{Command, Stdio};

    use temp_testdir::TempDir;

    pub struct TempGitRepository {
        tempdir: TempDir,
    }

    impl TempGitRepository {
        pub fn repository(&self) -> git2::Repository {
            let path = self.tempdir.join("interview-questions");
            git2::Repository::open(path).unwrap()
        }
    }

    impl Default for TempGitRepository {
        fn default() -> Self {
            let tempdir = TempDir::default();

            Command::new("git")
                .current_dir(&tempdir)
                .arg("clone")
                .args(["--depth", "1"])
                .arg("https://github.com/TabbyML/interview-questions")
                .stderr(Stdio::null())
                .stdout(Stdio::null())
                .status()
                .unwrap();

            Self { tempdir }
        }
    }
}
