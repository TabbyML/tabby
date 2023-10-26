use std::process::Command;

use anyhow::{anyhow, Result};
use tabby_common::config::{Config, RepositoryConfig};

trait ConfigExt {
    fn sync_repositories(&self) -> Result<()>;
}

impl ConfigExt for Config {
    fn sync_repositories(&self) -> Result<()> {
        for repository in self.repositories.iter() {
            if repository.is_local_dir() {
                if !repository.dir().exists() {
                    panic!("Directory {} does not exist", repository.dir().display());
                }
            } else {
                repository.sync()?;
            }
        }

        Ok(())
    }
}

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

pub fn sync_repositories(config: &Config) -> Result<()> {
    config.sync_repositories()
}
