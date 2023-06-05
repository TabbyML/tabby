use std::path::PathBuf;
use std::process::Command;

use tabby_common::{
    config::{Config, Repository},
    path::REPOSITORIES_DIR,
};

use filenamify::filenamify;

trait ConfigExt {
    fn sync_repositories(&self);
}

impl ConfigExt for Config {
    fn sync_repositories(&self) {
        for repository in self.repositories.iter() {
            repository.sync()
        }
    }
}

trait RepositoryExt {
    fn dir(&self) -> PathBuf;
    fn sync(&self);
}

impl RepositoryExt for Repository {
    fn dir(&self) -> PathBuf {
        REPOSITORIES_DIR.join(filenamify(&self.git_url))
    }

    fn sync(&self) {
        let dir = self.dir();
        let dir_string = dir.display().to_string();
        let status = if dir.exists() {
            Command::new("git")
                .current_dir(&dir)
                .arg("pull")
                .status()
                .expect("git could not be executed")
        } else {
            std::fs::create_dir_all(&dir);
            Command::new("git")
                .current_dir(dir.parent().unwrap())
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg(&self.git_url)
                .arg(dir)
                .status()
                .expect("git could not be executed")
        };

        if let Some(code) = status.code() {
            if code != 0 {
                panic!(
                    "Failed to pull remote '{}'\nConsider remove dir '{}' and retry",
                    &self.git_url, &dir_string
                );
            }
        }
    }
}

pub fn job_sync_repositories() {
    let config = Config::load();
    config.sync_repositories();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let config = Config {
            repositories: vec![Repository {
                git_url: "https://github.com/TabbyML/interview-questions".to_owned(),
            }],
        };

        config.sync_repositories();
    }
}
