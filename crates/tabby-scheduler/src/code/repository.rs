use std::{
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    process::Command,
};

use tabby_common::{config::RepositoryConfig, path::repositories_dir};
use tracing::{debug, warn};

trait RepositoryExt {
    fn sync(&self);
    fn last_sync(&self) -> PathBuf;
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
                finished = code == 0;
                if code != 0 {
                    warn!(
                        "Failed to clone `{}`. Please check your repository configuration.",
                        self.canonical_git_url()
                    );
                    fs::remove_dir_all(&dir).expect("Failed to remove directory");
                }
            }
        }

        if finished {
            debug!("Repository `{}` is up to date", dir.display());
            touch(&get_last_repository_sync_filepath(&self.dir()));
        }
    }

    fn last_sync(&self) -> PathBuf {
        self.dir()
            .join(".git")
            .join("tabby")
            .join("last_repository_sync")
    }
}

fn touch(path: &Path) {
    std::fs::create_dir_all(path.parent().expect("Failed to read parent"))
        .expect("Failed to create directory");
    OpenOptions::new()
        .write(true)
        .create(true)
        .open(path)
        .expect("Failed to touch file");
}

fn get_last_repository_sync_filepath(path: &Path) -> PathBuf {
    path.join(".git").join("tabby").join("last_repository_sync")
}

fn get_last_repository_sync_time(path: &Path) -> std::time::SystemTime {
    let filepath = get_last_repository_sync_filepath(path);
    if !filepath.exists() {
        touch(&filepath);
    }

    fs::metadata(get_last_repository_sync_filepath(&filepath))
        .expect("Failed to read metadata")
        .modified()
        .expect("Failed to read modified")
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

pub fn sync_repository(repository: &RepositoryConfig) {
    if repository.is_local_dir() {
        if !repository.dir().exists() {
            panic!("Directory {} does not exist", repository.dir().display());
        }
    } else {
        repository.sync();
    }

    garbage_collection();
}

fn garbage_collection() {
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
        } else if metadata.is_dir() {
            let mtime = get_last_repository_sync_time(&file.path());

            // if stale for 2 day, consider it as garbage.
            let is_garbage = mtime.elapsed().unwrap_or_default().as_secs() > 2 * 24 * 60 * 60;

            if is_garbage {
                warn!("An unrelated directory {:?} was found in repositories directory, It will now be removed...", file.path().display());
                fs::remove_dir_all(file.path()).unwrap_or_else(|_| {
                    panic!("Failed to remove directory {:?}", file.path().display())
                });
            }
        }
    }
}
