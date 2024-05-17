mod file_search;
mod serve_git;

pub use file_search::FileSearch;
pub use serve_git::ServeGit;

#[cfg(test)]
mod testutils {
    use std::process::{Command, Stdio};

    use temp_testdir::TempDir;

    pub struct TempGitRepository {
        tempdir: TempDir,
    }

    impl TempGitRepository {
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
