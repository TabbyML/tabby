use std::{
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

fn repository_file_key(repo_name: &str, path: &str) -> String {
    format!("{repo_name}:{path}")
}

impl IncrementalRepositoryStore {
    pub fn new() -> Result<Self> {
        Ok(Self {
            store: Store::new(Config::new(path::incremental_repository_store()))?,
        })
    }

    fn repository_versions_bucket(&self) -> Result<Bucket<String, String>> {
        Ok(self.store.bucket(Some(VERSIONS_BUCKET))?)
    }

    fn dataset_bucket(&self) -> Result<Bucket<String, Json<SourceFile>>> {
        Ok(self.store.bucket(Some(DATASET_BUCKET))?)
    }

    fn set_last_commit(&self, repo_path: String, commit_hash: String) -> Result<()> {
        self.repository_versions_bucket()?
            .set(&repo_path, &commit_hash)?;
        Ok(())
    }

    fn get_last_commit(&self, repo_path: String) -> Result<Option<String>> {
        Ok(self.repository_versions_bucket()?.get(&repo_path)?)
    }

    pub fn update_repository(&self, repository: &RepositoryConfig) -> Result<()> {
        let dir = repository.dir();
        let dir_str = dir.to_string_lossy().to_string();

        let old_version = self.get_last_commit(dir_str.clone())?;
        let current_version = get_git_commit(&dir)?;

        let Some(old_version) = old_version else {
            for file in repository.create_dataset() {
                self.update_source_file(file)?;
            }
            self.set_last_commit(dir_str.clone(), current_version)?;
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
        self.set_last_commit(dir_str.clone(), current_version)?;

        Ok(())
    }

    fn update_source_file(&self, file: SourceFile) -> Result<()> {
        let key = repository_file_key(&file.basedir, &file.filepath);
        self.dataset_bucket()?.set(&key, &Json(file))?;
        Ok(())
    }

    pub fn cached_source_files(&self) -> Result<impl Iterator<Item = SourceFile>> {
        Ok(self.dataset_bucket()?.iter().map(|entry| {
            entry
                .and_then(|item| item.value())
                .map(|Json(source_file)| source_file)
                .unwrap()
        }))
    }
}
