use std::{collections::HashSet, fs, path::Path, process::Command};

use tabby_common::{config::RepositoryConfig, path::repositories_dir};
use tracing::warn;

trait RepositoryExt {
    fn sync(&self);
}

impl RepositoryExt for RepositoryConfig {
    fn sync(&self) {
        let dir = self.dir();
        let mut finished = false;
        if dir.exists() {
            finished = pull_remote(dir.as_path());
        }

        if !finished {
            std::fs::create_dir_all(&dir)
                .unwrap_or_else(|_| panic!("Failed to create dir {}", dir.display()));
            let status = Command::new("git")
                .current_dir(dir.parent().expect("Must not be in root directory"))
                .arg("clone")
                .args(["--depth", "1"])
                .arg(&self.git_url)
                .arg(&dir)
                .status()
                .expect("Failed to read status");

            if let Some(code) = status.code() {
                if code != 0 {
                    warn!(
                        "Failed to clone `{}`. Please check your repository configuration.",
                        &self.git_url
                    );
                    fs::remove_dir_all(&dir).expect("Failed to remove directory");
                }
            }
        }
    }
}

fn pull_remote(path: &Path) -> bool {
    let status = Command::new("git")
        .current_dir(path)
        .arg("pull")
        .status()
        .expect("Failed to read status");

    if let Some(code) = status.code() {
        if code != 0 {
            warn!(
                "Failed to pull remote for `{:?}`, It will now be removed...",
                path
            );
            fs::remove_dir_all(path).expect("Failed to remove directory");
            return false;
        }
    };

    true
}

pub fn sync_repositories(repositories: &[RepositoryConfig]) {
    // Ensure repositories_dir exist.
    std::fs::create_dir_all(repositories_dir()).expect("Failed to create repositories directory");

    let mut names = HashSet::new();
    for repository in repositories {
        names.insert(repository.dir());
        if repository.is_local_dir() {
            if !repository.dir().exists() {
                panic!("Directory {} does not exist", repository.dir().display());
            }
        } else {
            repository.sync();
        }
    }

    for file in fs::read_dir(repositories_dir())
        .expect("Failed to read repository dir")
        .filter_map(Result::ok)
    {
        let metadata = file.metadata().expect("Failed to read metadata");
        let filename = file.file_name();
        if metadata.is_file() {
            warn!("An unrelated file {:?} was found in repositories directory, It will now be removed...", filename);
            // There shouldn't be any files under repositories dir.
            fs::remove_file(file.path())
                .unwrap_or_else(|_| panic!("Failed to remove file {:?}", filename))
        } else if metadata.is_dir() && !names.contains(&file.path()) {
            warn!("An unrelated directory {:?} was found in repositories directory, It will now be removed...", file.path().display());
            fs::remove_dir_all(file.path()).unwrap_or_else(|_| {
                panic!("Failed to remove directory {:?}", file.path().display())
            });
        }
    }
}
