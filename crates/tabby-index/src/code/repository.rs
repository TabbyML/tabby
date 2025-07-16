use std::{
    collections::HashSet,
    fs::{self},
    path::Path,
};

use anyhow::{bail, Context};
use git2::{
    build::{CheckoutBuilder, RepoBuilder},
    Cred, FetchOptions, RemoteCallbacks, Repository,
};
use tabby_common::{config::SshKeyPair, path::repositories_dir};
use tracing::warn;

use super::CodeRepository;

trait RepositoryExt {
    fn sync(&self) -> anyhow::Result<String>;
}

impl RepositoryExt for CodeRepository {
    // sync clones the repository if it doesn't exist, otherwise it pulls the remote.
    // and returns the git commit sha256.
    fn sync(&self) -> anyhow::Result<String> {
        let dir = self.dir();
        if dir.exists() {
            pull_repo(self, &dir).with_context(|| {
                format!(
                    "failed to pull repo {} at {}",
                    self.canonical_git_url(),
                    self.dir().display(),
                )
            })?;
        } else {
            clone_repo(self, &dir).with_context(|| {
                format!(
                    "failed to clone repo {} into {}",
                    self.canonical_git_url(),
                    self.dir().display(),
                )
            })?;
        }

        get_commit_sha(self)
    }
}

fn get_commit_sha(repository: &CodeRepository) -> anyhow::Result<String> {
    let repo = git2::Repository::open(repository.dir())?;
    let head = repo.head()?;
    let commit = head.peel_to_commit()?;
    Ok(commit.id().to_string())
}

fn get_fetch_options<'r>(repo: &'r CodeRepository) -> FetchOptions<'r> {
    let mut callbacks = RemoteCallbacks::new();
    if let Some(keypair) = &repo.ssh_key {
        match keypair {
            SshKeyPair::Memory(public_key, private_key) => {
                callbacks.credentials(move |_url, username_from_url, _allowed_types| {
                    Cred::ssh_key_from_memory(
                        username_from_url.unwrap(),
                        public_key.as_deref(),
                        private_key,
                        None,
                    )
                });
            }
            SshKeyPair::Paths(public_key, private_key) => {
                callbacks.credentials(move |_url, username_from_url, _allowed_types| {
                    Cred::ssh_key(
                        username_from_url.unwrap(),
                        public_key.as_deref(),
                        private_key,
                        None,
                    )
                });
            }
        }
    }

    let mut fo = FetchOptions::new();
    fo.remote_callbacks(callbacks);

    fo
}

fn pull_repo(code_repo: &CodeRepository, path: &Path) -> anyhow::Result<()> {
    let repo = Repository::open(path)?;

    let mut remote = repo.find_remote("origin")?;

    let mut fo = get_fetch_options(code_repo);

    remote.fetch(&["refs/heads/*:refs/heads/*"], Some(&mut fo), None)?;

    repo.checkout_head(Some(CheckoutBuilder::default().force()))?;

    Ok(())
}

fn do_clone_repo(code_repo: &CodeRepository, path: &Path) -> anyhow::Result<()> {
    if code_repo.ssh_key.is_some() {
        let mut builder = RepoBuilder::new();

        let fo = get_fetch_options(code_repo);
        builder.fetch_options(fo);

        builder.clone(&code_repo.git_url, path)?;
    } else {
        Repository::clone(&code_repo.git_url, path)?;
    }

    Ok(())
}

fn clone_repo(code_repo: &CodeRepository, path: &Path) -> anyhow::Result<()> {
    do_clone_repo(code_repo, path).map_err(|err| {
        warn!("Failed to clone repository: {}", err);
        if path.exists() {
            fs::remove_dir_all(path).expect("Failed to remove cloned repository");
        }
        err
    })
}

pub fn sync_repository(repository: &CodeRepository) -> anyhow::Result<String> {
    if repository.is_local_dir() {
        if !repository.dir().exists() {
            bail!("Directory {} does not exist", repository.dir().display());
        }
        get_commit_sha(repository)
    } else {
        repository.sync()
    }
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

#[cfg(test)]
mod test {
    use std::env;

    use tabby_common::config::SshKeyPair;

    use super::{CodeRepository, RepositoryExt};

    #[test]
    fn test_public_repo_clone() -> anyhow::Result<()> {
        let repo = CodeRepository::new("https://github.com/TabbyML/tabby/", "1");
        repo.sync()?;
        Ok(())
    }

    #[test]
    fn test_private_repo_clone_keys_from_path() -> anyhow::Result<()> {
        if let Ok(repo_url) = env::var("TABBY_TEST_PRIVATE_REPO_FOR_PATH_KEYS") {
            let mut repo = CodeRepository::new(&repo_url, "2");
            repo.with_ssh_key(&SshKeyPair::Paths(
                env::var("TABBY_TEST_PUBLIC_KEY_PATH").map(|s| s.into())
                    .ok(),
                env::var("TABBY_TEST_PRIVATE_KEY_PATH")?.into(),
            ));
            repo.sync()?;
        }
        Ok(())
    }

    #[test]
    fn test_private_repo_clone_keys_from_content() -> anyhow::Result<()> {
        if let Ok(repo_url) = &env::var("TABBY_TEST_PRIVATE_REPO_FOR_MEM_KEYS") {
            let mut repo = CodeRepository::new(repo_url, "3");
            repo.with_ssh_key(&SshKeyPair::Memory(
                env::var("TABBY_TEST_PUBLIC_KEY")
                    .ok(),
                env::var("TABBY_TEST_PRIVATE_KEY")?,
            ));
            repo.sync()?;
        }
        Ok(())
    }
}
