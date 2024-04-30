use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Result;
use kv::{Bucket, Config, Json, Store, Transaction, TransactionError};
use serde::{Deserialize, Serialize};
use tabby_common::{config::RepositoryConfig, path, SourceFile};

use crate::{
    code::CodeIntelligence,
    dataset::{create_source_file, RepositoryExt},
};

const META_KEY: &str = "meta";
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

fn meta_key(git_url: impl AsRef<str>) -> String {
    format!("{META_KEY}:{}", git_url.as_ref())
}

pub struct RepositoryStore {
    store: Store,
}

#[derive(Serialize, Deserialize, Default)]
struct RepositoryMeta {
    last_sync_commit: Option<String>,
    last_index_commit: Option<String>,
}

impl RepositoryStore {
    pub fn new() -> Result<Self> {
        Ok(Self {
            store: Store::new(Config::new(path::repository_store()))?,
        })
    }

    fn meta_bucket(&self) -> Bucket<String, Json<RepositoryMeta>> {
        self.store.bucket(None).unwrap()
    }

    fn dataset_bucket(&self, repository: &RepositoryConfig) -> Bucket<String, Json<SourceFile>> {
        self.store
            .bucket(Some(&format!(
                "{DATASET_BUCKET}:{}",
                repository.canonical_git_url()
            )))
            .unwrap()
    }

    fn set_last_sync_commit(
        &self,
        transaction: &Transaction<String, Json<RepositoryMeta>>,
        repository: &RepositoryConfig,
        commit_hash: String,
    ) -> Result<()> {
        let mut meta = self.get_meta(transaction, repository)?;
        meta.last_sync_commit = Some(commit_hash);
        self.meta_bucket()
            .set(&meta_key(repository.canonical_git_url()), &Json(meta))?;
        Ok(())
    }

    fn get_meta(
        &self,
        transaction: &Transaction<String, Json<RepositoryMeta>>,
        repository: &RepositoryConfig,
    ) -> Result<RepositoryMeta> {
        Ok(transaction
            .get(&meta_key(&repository.canonical_git_url()))?
            .map(|Json(meta)| meta)
            .unwrap_or_default())
    }

    pub fn update_repository(&self, repository: &RepositoryConfig) {
        let dir = repository.dir();

        self.meta_bucket()
            .transaction2(
                &self.dataset_bucket(repository),
                |meta_bucket, repo_bucket| {
                    let old_version = self
                        .get_meta(&meta_bucket, repository)
                        .unwrap()
                        .last_sync_commit;
                    let current_version = get_git_commit(&dir).unwrap();

                    let Some(old_version) = old_version else {
                        for file in repository.create_dataset() {
                            self.update_source_file(&repo_bucket, file).unwrap();
                        }
                        self.set_last_sync_commit(&meta_bucket, repository, current_version)
                            .unwrap();
                        return Ok::<_, TransactionError<kv::Error>>(());
                    };

                    let files_diff = get_changed_files(&dir, old_version).unwrap();
                    let mut code = CodeIntelligence::default();
                    for file in files_diff {
                        let path = dir.join(&file);
                        let Some(source_file) = create_source_file(repository, &path, &mut code)
                        else {
                            self.remove_source_file(&repo_bucket, file).unwrap();
                            continue;
                        };
                        self.update_source_file(&repo_bucket, source_file).unwrap();
                    }
                    self.set_last_sync_commit(&meta_bucket, repository, current_version)
                        .unwrap();
                    Ok(())
                },
            )
            .unwrap()
    }

    fn update_source_file(
        &self,
        transaction: &Transaction<String, Json<SourceFile>>,
        file: SourceFile,
    ) -> Result<()> {
        transaction.set(&file.git_url.clone(), &Json(file))?;
        Ok(())
    }

    fn remove_source_file(
        &self,
        transaction: &Transaction<String, Json<SourceFile>>,
        path: String,
    ) -> Result<()> {
        transaction.remove(&path)?;
        Ok(())
    }

    pub fn cached_source_files(&self) -> impl Iterator<Item = SourceFile> + '_ {
        self.store
            .buckets()
            .into_iter()
            .flat_map(|bucket_name| {
                self.store
                    .bucket::<String, Json<SourceFile>>(Some(&bucket_name))
                    .unwrap()
                    .iter()
            })
            .map(|item| item.unwrap().value().unwrap())
            .map(|Json(source_file)| source_file)
    }

    pub fn retain_from(&self, configs: &[RepositoryConfig]) {
        let added_repositories: HashSet<_> = configs
            .iter()
            .map(|config| config.canonical_git_url())
            .collect();

        for bucket in self.store.buckets() {
            if !added_repositories.contains(&bucket) {
                self.store.drop_bucket(&bucket).unwrap();
                self.meta_bucket().remove(&meta_key(&bucket)).unwrap();
            }
        }
    }
}
