mod file_search;
mod serve_git;

use std::path::Path;

use axum::{
    body::Body,
    http::{Response, StatusCode},
};
use file_search::GitFileSearch;

pub struct GitReadOnly {}

impl GitReadOnly {
    pub async fn search_files(
        root: &Path,
        commit: Option<&str>,
        pattern: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<GitFileSearch>> {
        file_search::search(git2::Repository::open(root)?, commit, pattern, limit).await
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
        Ok(refs.filter_map(|r| r.ok())
            .map(|r| r.name().unwrap().to_string())
            // Filter out remote refs
            .filter(|r| !r.starts_with("refs/remotes/"))
            .collect())
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
