use std::{collections::HashSet, fs, process::Command};

use anyhow::{anyhow, Result};
use tabby_common::{config::RepositoryConfig, path::repositories_dir};
use tracing::warn;

trait RepositoryExt {
    fn sync(&self) -> Result<()>;
}

impl RepositoryExt for RepositoryConfig {
    fn sync(&self) -> Result<()> {
        let dir = self.dir();
        let dir_string = dir.display().to_string();
        let status = if dir.exists() {
            Command::new("git").current_dir(&dir).arg("pull").status()
        } else {
            std::fs::create_dir_all(&dir)
                .unwrap_or_else(|_| panic!("Failed to create dir {}", dir_string));
            Command::new("git")
                .current_dir(dir.parent().unwrap())
                .arg("clone")
                .arg(&self.git_url)
                .arg(dir)
                .status()
        };

        if let Some(code) = status?.code() {
            if code != 0 {
                return Err(anyhow!(
                    "Failed to pull remote '{}'. Consider remove dir '{}' and retry",
                    &self.git_url,
                    &dir_string
                ));
            }
        }

        Ok(())
    }
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
