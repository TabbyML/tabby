use std::{
    collections::HashSet,
    fs::{self},
    path::Path,
    process::Command,
};

use anyhow::bail;
use tabby_common::path::repositories_dir;
use tracing::warn;

use super::CodeRepository;

trait RepositoryExt {
    fn sync(&self) -> anyhow::Result<()>;
}

impl RepositoryExt for CodeRepository {
    fn sync(&self) -> anyhow::Result<()> {
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
                .arg(&self.git_url)
                .arg(&dir)
                .status()
                .unwrap_or_else(|_| panic!("Failed to clone into dir {}", dir.display()));

            if let Some(code) = status.code() {
                if code != 0 {
                    warn!(
                        "Failed to clone `{}`. Please check your repository configuration.",
                        self.canonical_git_url()
                    );
                    fs::remove_dir_all(&dir).expect("Failed to remove directory");

                    bail!("Failed to clone `{}`", self.canonical_git_url());
                }
            }
        }

        Ok(())
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
                "Failed to pull remote for `{:?}`, please check your repository configuration...",
                path
            );
            return false;
        }
    };

    true
}

pub fn sync_repository(repository: &CodeRepository) -> anyhow::Result<()> {
    if repository.is_local_dir() {
        if !repository.dir().exists() {
            panic!("Directory {} does not exist", repository.dir().display());
        }
    } else {
        repository.sync()?;
    }

    Ok(())
}

pub fn garbage_collection(repositories: &[CodeRepository]) {
    let names = repositories.iter().map(|r| r.dir()).collect::<HashSet<_>>();

    let Ok(dir) = fs::read_dir(repositories_dir()) else {
        return;
    };

    for file in dir.filter_map(Result::ok) {
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
