use std::{collections::HashSet, fs, path::Path, process::Command};

use anyhow::{anyhow, Result};
use tabby_common::{config::RepositoryConfig, path::repositories_dir};
use tracing::warn;

trait RepositoryExt {
    fn sync(&self) -> Result<()>;
}

impl RepositoryExt for RepositoryConfig {
    fn sync(&self) -> Result<()> {
        let dir = self.dir();
        let mut finished = false;
        if dir.exists() {
            finished = pull_remote(dir.as_path())?;
        }

        if !finished {
            std::fs::create_dir_all(&dir)
                .unwrap_or_else(|_| panic!("Failed to create dir {}", dir.display().to_string()));
            let status = Command::new("git")
                .current_dir(dir.parent().unwrap())
                .arg("clone")
                .arg(&self.git_url)
                .arg(dir)
                .status()?;

            if let Some(code) = status.code() {
                if code != 0 {
                    return Err(anyhow!(
                        "Failed to pull remote '{}'. Please check your repository configuration",
                        &self.git_url,
                    ));
                }
            }
        }

        Ok(())
    }
}

fn pull_remote(path: &Path) -> std::io::Result<bool> {
    let status = Command::new("git").current_dir(path).arg("pull").status()?;

    if let Some(code) = status.code() {
        if code != 0 {
            warn!(
                "Failed to pull remote for `{:?}`, It will now be removed...",
                path
            );
            fs::remove_dir_all(path)?;
            return Ok(false);
        }
    };

    Ok(true)
}

pub fn sync_repositories(repositories: &[RepositoryConfig]) -> Result<()> {
    // Ensure repositories_dir exist.
    std::fs::create_dir_all(repositories_dir())?;

    let mut names = HashSet::new();
    for repository in repositories {
        names.insert(repository.name());
        if repository.is_local_dir() {
            if !repository.dir().exists() {
                panic!("Directory {} does not exist", repository.dir().display());
            }
        } else {
            repository.sync()?;
        }
    }

    for file in fs::read_dir(repositories_dir())?.filter_map(Result::ok) {
        let metadata = file.metadata()?;
        let filename = file.file_name();
        if metadata.is_file() {
            warn!("An unrelated file {:?} was found in repositories directory, It will now be removed...", filename);
            // There shouldn't be any files under repositories dir.
            fs::remove_file(file.path())?;
        } else if metadata.is_dir() {
            let filename = filename.to_str().ok_or(anyhow!("Invalid file name"))?;
            if !names.contains(filename) {
                warn!("An unrelated directory {:?} was found in repositories directory, It will now be removed...", filename);
                fs::remove_dir_all(file.path())?;
            }
        }
    }

    Ok(())
}
