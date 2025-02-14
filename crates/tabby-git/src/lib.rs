mod commit;
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

pub use commit::{stream_commits, Commit};
pub use grep::{GrepFile, GrepLine, GrepSubMatch, GrepTextOrBase64};

pub async fn search_files(
    root: &Path,
    rev: Option<&str>,
    pattern: &str,
    limit: usize,
) -> anyhow::Result<Vec<GitFileSearch>> {
    file_search::search(git2::Repository::open(root)?, rev, pattern, limit).await
}

pub struct ListFile {
    pub files: Vec<GitFileSearch>,
    pub truncated: bool,
}

pub async fn list_files(
    root: &Path,
    rev: Option<&str>,
    limit: Option<usize>,
) -> anyhow::Result<ListFile> {
    let (files, truncated) = file_search::list(git2::Repository::open(root)?, rev, limit).await?;
    Ok(ListFile { files, truncated })
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

pub struct GitReference {
    pub name: String,
    pub commit: String,
}

pub fn list_refs(root: &Path) -> anyhow::Result<Vec<GitReference>> {
    let repository = git2::Repository::open(root)?;
    let refs = repository.references()?;
    Ok(refs
        .filter_map(|r| r.ok())
        .filter_map(|r| {
            let name = r.name()?.to_string();
            let commit = r.target()?.to_string();
            Some(GitReference { name, commit })
        })
        // Filter out remote refs
        .filter(|r| !r.name.starts_with("refs/remotes/"))
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
            git2::Repository::open(self.path()).unwrap()
        }

        pub fn path(&self) -> std::path::PathBuf {
            self.tempdir.join("interview-questions")
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

#[cfg(test)]
mod tests {
    use crate::{list_refs, testutils::TempGitRepository};

    #[test]
    fn test_list_refs() {
        let root = TempGitRepository::default();
        let refs = list_refs(&root.path()).unwrap();
        assert_eq!(refs.len(), 1);
    }
}
