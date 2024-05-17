mod file_search;
mod serve_git;

use axum::{
    body::Body,
    http::{Response, StatusCode},
};
use file_search::GitFileSearch;

pub struct GitReadOnly {
    repository: git2::Repository,
}

impl GitReadOnly {
    pub fn new(path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(Self {
            repository: git2::Repository::open(path)?,
        })
    }

    pub fn search_files(
        &self,
        commit: Option<&str>,
        pattern: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<GitFileSearch>> {
        file_search::search(&self.repository, commit, pattern, limit)
    }

    pub fn serve_file(
        &self,
        commit: Option<&str>,
        path: Option<&str>,
    ) -> std::result::Result<Response<Body>, StatusCode> {
        serve_git::serve(&self.repository, commit, path)
    }
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
