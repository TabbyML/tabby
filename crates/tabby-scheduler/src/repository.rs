use std::process::Command;

use tabby_common::config::{Config, Repository};

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
    fn sync(&self);
}

impl RepositoryExt for Repository {
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
            std::fs::create_dir_all(&dir)
                .unwrap_or_else(|_| panic!("Failed to create dir {}", dir_string));
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

pub fn sync_repositories(config: &Config) {
    config.sync_repositories();
}
