use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Result;
use kv::{Bucket, Config, Json, Store};
use tabby_common::{config::RepositoryConfig, SourceFile};

use crate::{
    code::CodeIntelligence,
    dataset::{create_source_file, RepositoryExt},
    path,
};

pub struct IncrementalRepositoryStore {
    store: Store,
}

const VERSIONS_BUCKET: &str = "versions";
const DATASET_BUCKET: &str = "dataset";

fn cmd_stdout(pwd: &Path, cmd: &str, args: &[&str]) -> Result<String> {
    Ok(String::from_utf8(
        Command::new(cmd)
            .current_dir(pwd)
            .args(args)
            .spawn()?
            .wait_with_output()?
            .stdout,
    )?)
}

fn get_git_commit(path: &Path) -> Result<String> {
    cmd_stdout(path, "git", &["rev-parse", "HEAD"])
}

fn get_changed_files(path: &Path, since_commit: String) -> Result<Vec<String>> {
    cmd_stdout(path, "git", &["diff", "--name-only", &since_commit])
        .map(|s| s.lines().map(|line| line.to_owned()).collect())
}

fn repository_file_key(base_dir: &str, path: &str) -> String {
    format!("{base_dir}:{path}")
}

fn parse_key_base_dir(key: &str) -> &str {
    key.split_once(':').unwrap().0
}

impl IncrementalRepositoryStore {
    pub fn new() -> Result<Self> {
        Ok(Self {
            store: Store::new(Config::new(path::incremental_repository_store()))?,
        })
    }

    fn repository_versions_bucket(&self) -> Bucket<String, String> {
        self.store.bucket(Some(VERSIONS_BUCKET)).unwrap()
    }

    fn dataset_bucket(&self) -> Bucket<String, Json<SourceFile>> {
        self.store.bucket(Some(DATASET_BUCKET)).unwrap()
    }

    fn set_last_commit(&self, canonical_git_url: String, commit_hash: String) -> Result<()> {
        self.repository_versions_bucket()
            .set(&canonical_git_url, &commit_hash)?;
        Ok(())
    }

    fn get_last_commit(&self, canonical_git_url: String) -> Result<Option<String>> {
        Ok(self.repository_versions_bucket().get(&canonical_git_url)?)
    }

    pub fn update_repository(&self, repository: &RepositoryConfig) -> Result<()> {
        let dir = repository.dir();
        let canonical_git_url = repository.canonical_git_url();

        let old_version = self.get_last_commit(canonical_git_url.clone())?;
        let current_version = get_git_commit(&dir)?;

        let Some(old_version) = old_version else {
            for file in repository.create_dataset() {
                self.update_source_file(file)?;
            }
            self.set_last_commit(canonical_git_url.clone(), current_version)?;
            return Ok(());
        };

        let files_diff = get_changed_files(&dir, old_version)?;
        let mut code = CodeIntelligence::default();
        for file in files_diff {
            let Some(source_file) = create_source_file(repository, &PathBuf::from(file), &mut code)
            else {
                continue;
            };
            self.update_source_file(source_file)?;
        }
        self.set_last_commit(canonical_git_url.clone(), current_version)?;

        Ok(())
    }

    fn update_source_file(&self, file: SourceFile) -> Result<()> {
        // git_url is equal to canonical_git_url()
        let key = repository_file_key(&file.git_url, &file.filepath);
        self.dataset_bucket().set(&key, &Json(file))?;
        Ok(())
    }

    pub fn cached_source_files(&self) -> impl Iterator<Item = SourceFile> {
        self.dataset_bucket().iter().map(|entry| {
            entry
                .and_then(|item| item.value())
                .map(|Json(source_file)| source_file)
                .unwrap()
        })
    }

    pub fn retain_from(&self, configs: &[RepositoryConfig]) {
        let added_repositories: HashSet<_> = configs
            .iter()
            .map(|config| config.canonical_git_url())
            .collect();

        let dataset_bucket = self.dataset_bucket();
        let to_remove: Vec<_> = dataset_bucket
            .iter()
            .map(|item| item.unwrap().key::<String>().unwrap())
            .filter(|key| !added_repositories.contains(parse_key_base_dir(key)))
            .collect();

        for key in to_remove {
            dataset_bucket.remove(&key).unwrap();
        }

        let versions_bucket = self.repository_versions_bucket();
        let versions_to_remove: Vec<_> = versions_bucket
            .iter()
            .map(|item| item.unwrap().key::<String>().unwrap())
            .filter(|key| !added_repositories.contains(key))
            .collect();

        for key in versions_to_remove {
            versions_bucket.remove(&key).unwrap();
        }
    }
}
