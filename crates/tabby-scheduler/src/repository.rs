use std::{path::PathBuf, process::Command};

use filenamify::filenamify;
use tabby_common::{
    config::{Config, Repository},
    path::repositories_dir,
};

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
        repositories_dir().join(filenamify(&self.git_url))
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

pub fn sync_repositories() {
    let config = Config::load();
    config.sync_repositories();
}

#[cfg(test)]
mod tests {
    use tabby_common::{
        config::{Config, Repository},
        path::set_tabby_root,
    };
    use temp_testdir::*;

    use super::*;

    #[test]
    fn it_works() {
        set_tabby_root(TempDir::default().to_path_buf());

        let config = Config {
            repositories: vec![Repository {
                git_url: "https://github.com/TabbyML/interview-questions".to_owned(),
            }],
        };

        config.save();
        sync_repositories();
    }
}
