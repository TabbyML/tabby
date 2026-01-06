use std::{
    collections::HashSet,
    fs::{self},
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
    // sync clones the repository if it doesn't exist, otherwise it pulls the remote.
    fn sync(&self) -> anyhow::Result<()> {
        let dir = self.dir();
        if !dir.exists() {
            logkit::info!("Cloning repository {}", self.canonical_git_url());
            std::fs::create_dir_all(&dir)?;
            let status = Command::new("git")
                .current_dir(dir.parent().expect("Must not be in root directory"))
                .arg("clone")
                .arg(&self.git_url)
                .arg(&dir)
                .status()?;

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

        let mut failed = vec![];
        for ref_name in &self.git_refs {
            logkit::info!("Fetching reference: {}", ref_name);

            // get the current branch name without refs/ prefix
            let output = Command::new("git")
                .current_dir(&dir)
                .arg("symbolic-ref")
                .arg("--short")
                .arg("HEAD")
                .output()
                .ok();

            let current_branch = output
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string());

            let status = if current_branch.as_deref() == Some(ref_name) {
                Command::new("git")
                    .current_dir(&dir)
                    .arg("pull")
                    .arg("origin")
                    .arg(ref_name)
                    .status()
            } else {
                // Use `git fetch origin +ref:ref` to create or update the local branch from the remote.
                // The + ensures that the local branch is updated (forced) even if it's not a fast-forward,
                //   and it creates the branch if it doesn't exist locally.
                Command::new("git")
                    .current_dir(&dir)
                    .arg("fetch")
                    .arg("origin")
                    .arg(format!("+{}:{}", ref_name, ref_name))
                    .status()
            };
            match status {
                Ok(exit_status) => {
                    if !exit_status.success() {
                        failed.push(ref_name.to_owned());
                        logkit::error!("Failed to fetch remote branch {}", ref_name);
                    }
                }
                Err(e) => {
                    failed.push(ref_name.to_owned());
                    logkit::error!("Failed to fetch remote branch {}: {}", ref_name, e);
                }
            }
        }

        if !failed.is_empty() {
            if failed.len() == self.git_refs.len() {
                logkit::error!("Failed to fetch all branches.");
                bail!("Failed to fetched all branches.");
            }

            logkit::warn!(
                "Failed to fetch {} out of {} branches, the others were indexed successfully.",
                failed.len(),
                self.git_refs.len()
            );
        }
        Ok(())
    }
}

pub fn sync_repository(repository: &CodeRepository) -> anyhow::Result<()> {
    if repository.is_local_dir() {
        if !repository.dir().exists() {
            bail!("Directory {} does not exist", repository.dir().display());
        }
    } else {
        repository.sync()?;
    }

    Ok(())
}

/// Resolve commits for the given repository.
///
/// This function inspects the git repository and returns a list of (ref_name, commit_sha)
/// pairs that should be indexed.
/// If no specific refs are configured, it defaults to HEAD.
pub fn resolve_commits(repository: &CodeRepository) -> Vec<(String, String)> {
    let repo = match git2::Repository::open(repository.dir()) {
        Ok(repo) => repo,
        Err(e) => {
            logkit::error!(
                "failed to open repo {}: {}",
                repository.canonical_git_url(),
                e
            );
            return vec![];
        }
    };

    let mut commits = Vec::new();

    // if no refs specified, use the default branch and commits directly
    if repository.git_refs.is_empty() {
        if let Ok(head) = repo.head() {
            if let Ok(commit) = head.peel_to_commit() {
                commits.push((
                    head.name().unwrap_or("HEAD").to_string(),
                    commit.id().to_string(),
                ));
            }
        }
        return commits;
    }

    for ref_name in &repository.git_refs {
        let reference = match repo
            .find_reference(&format!("refs/heads/{}", ref_name))
            .or_else(|_| repo.find_reference(&format!("refs/tags/{}", ref_name)))
            .or_else(|_| repo.find_reference(ref_name))
        {
            Ok(reference) => reference,
            Err(e) => {
                logkit::error!("failed to find ref {}: {}", ref_name, e);
                continue;
            }
        };

        let commit = match reference.peel_to_commit() {
            Ok(commit) => commit,
            Err(e) => {
                logkit::error!("failed to get commit for ref {}: {}", ref_name, e);
                continue;
            }
        };
        commits.push((ref_name.clone(), commit.id().to_string()));
    }
    commits
}

pub fn checkout(repository: &CodeRepository, branch: &str) -> anyhow::Result<()> {
    let repo = git2::Repository::open(repository.dir())?;
    let reference = repo
        .find_reference(&format!("refs/heads/{}", branch))
        .or_else(|_| repo.find_reference(&format!("refs/tags/{}", branch)))
        .or_else(|_| repo.find_reference(branch))?;

    let mut checkout_builder = git2::build::CheckoutBuilder::new();
    checkout_builder.force();

    repo.set_head(reference.name().unwrap())?;
    repo.checkout_head(Some(&mut checkout_builder))?;
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
                .unwrap_or_else(|_| panic!("Failed to remove file {filename:?}"))
        } else if metadata.is_dir() && !names.contains(&file.path()) {
            warn!("An unrelated directory {:?} was found in repositories directory, It will now be removed...", file.path().display());
            fs::remove_dir_all(file.path()).unwrap_or_else(|_| {
                panic!("Failed to remove directory {:?}", file.path().display())
            });
        }
    }
}
