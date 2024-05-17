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

    pub fn list_references(root: &Path) -> anyhow::Result<Vec<String>> {
        let repository = git2::Repository::open(root)?;
        let mut refs = Vec::new();
        for reference in repository.references()? {
            let reference = reference?;
            let name = reference.name().unwrap();
            refs.push(name.to_string());
        }
        Ok(refs)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_refs() {
        let root = testutils::TempGitRepository::default();
        let refs = GitReadOnly::list_references(root.repository().workdir().unwrap()).unwrap();
        assert_eq!(refs.len(), 3);
        assert_eq!(refs[0], "refs/heads/main");
        assert_eq!(refs[1], "refs/remotes/origin/HEAD");
        assert_eq!(refs[3], "refs/remotes/origin/main");
    }
}