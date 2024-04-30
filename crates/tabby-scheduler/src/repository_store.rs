use std::{collections::HashSet, path::Path, process::Command};

use anyhow::Result;
use kv::{Bucket, Config, Json, Store, Transaction, TransactionError};
use serde::{Deserialize, Serialize};
use tabby_common::{config::RepositoryConfig, path, SourceFile};

use crate::{
    code::CodeIntelligence,
    dataset::{create_source_file, RepositoryExt},
};

const META_KEY: &str = "meta";
const DATASET_BUCKET_PREFIX: &str = "dataset";

fn cmd_stdout(pwd: &Path, cmd: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(cmd).current_dir(pwd).args(args).output()?;

    Ok(String::from_utf8(output.stdout)?.trim().to_string())
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

fn dataset_bucket_key(git_url: impl AsRef<str>) -> String {
    format!("{DATASET_BUCKET_PREFIX}:{}", git_url.as_ref())
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
    pub fn new() -> Self {
        Self {
            store: Store::new(Config::new(path::repository_store()))
                .expect("Failed to create repository store"),
        }
    }

    fn meta_bucket(&self) -> Bucket<String, Json<RepositoryMeta>> {
        self.store
            .bucket(None)
            .expect("Could not access meta bucket")
    }

    fn dataset_bucket(&self, repository: &RepositoryConfig) -> Bucket<String, Json<SourceFile>> {
        self.store
            .bucket(Some(&dataset_bucket_key(repository.canonical_git_url())))
            .expect("Could not access dataset bucket")
    }

    fn set_last_sync_commit(
        &self,
        transaction: &Transaction<String, Json<RepositoryMeta>>,
        repository: &RepositoryConfig,
        commit_hash: String,
    ) {
        let mut meta = self.get_meta(transaction, repository);
        meta.last_sync_commit = Some(commit_hash);
        transaction
            .set(&meta_key(repository.canonical_git_url()), &Json(meta))
            .expect("Failed to update synced version for repository");
    }

    fn get_meta(
        &self,
        transaction: &Transaction<String, Json<RepositoryMeta>>,
        repository: &RepositoryConfig,
    ) -> RepositoryMeta {
        transaction
            .get(&meta_key(repository.canonical_git_url()))
            .expect("Failed to access repository meta")
            .map(|Json(meta)| meta)
            .unwrap_or_default()
    }

    pub fn update_dataset(&self, repositories: &[RepositoryConfig]) {
        for repository in repositories {
            self.sync_repository(repository);
        }
        self.retain_from(repositories);
    }

    fn sync_repository(&self, repository: &RepositoryConfig) {
        let dir = repository.dir();

        self.meta_bucket()
            .transaction2(
                &self.dataset_bucket(repository),
                |meta_bucket, repo_bucket| {
                    let old_version = self.get_meta(&meta_bucket, repository).last_sync_commit;
                    let current_version = get_git_commit(&dir).unwrap();

                    let Some(old_version) = old_version else {
                        self.sync_repository_from_scratch(
                            &meta_bucket,
                            &repo_bucket,
                            current_version,
                            repository,
                        );
                        return Ok::<_, TransactionError<kv::Error>>(());
                    };

                    let files_diff = get_changed_files(&dir, old_version).unwrap();
                    let mut code = CodeIntelligence::default();
                    for file in files_diff {
                        let path = dir.join(&file);

                        if let Some(source_file) = create_source_file(repository, &path, &mut code)
                        {
                            // File exists and was either created or updated
                            repo_bucket
                                .set(&source_file.git_url.clone(), &Json(source_file))
                                .expect("Failed to update source file");
                        } else {
                            // File has been removed
                            repo_bucket
                                .remove(&file)
                                .expect("Failed to remove source file");
                        }
                    }
                    self.set_last_sync_commit(&meta_bucket, repository, current_version);
                    Ok(())
                },
            )
            .unwrap()
    }

    fn sync_repository_from_scratch(
        &self,
        meta_bucket: &Transaction<String, Json<RepositoryMeta>>,
        repo_bucket: &Transaction<String, Json<SourceFile>>,
        current_version: String,
        repository: &RepositoryConfig,
    ) {
        for file in repository.create_dataset() {
            repo_bucket
                .set(&file.git_url.clone(), &Json(file))
                .expect("Failed to update source file");
        }
        self.set_last_sync_commit(meta_bucket, repository, current_version);
    }

    pub fn cached_source_files(&self) -> impl Iterator<Item = SourceFile> + '_ {
        self.store
            .buckets()
            .into_iter()
            .filter(|bucket_name| bucket_name.starts_with(DATASET_BUCKET_PREFIX))
            .flat_map(|bucket_name| {
                self.store
                    .bucket::<String, Json<SourceFile>>(Some(&bucket_name))
                    .unwrap()
                    .iter()
            })
            .map(|item| item.unwrap().value().unwrap())
            .map(|Json(source_file)| source_file)
    }

    fn retain_from(&self, configs: &[RepositoryConfig]) {
        let added_repositories: HashSet<_> = configs
            .iter()
            .map(|config| dataset_bucket_key(config.canonical_git_url()))
            .collect();

        for bucket in self.store.buckets() {
            if bucket.starts_with(DATASET_BUCKET_PREFIX) && !added_repositories.contains(&bucket) {
                self.store.drop_bucket(&bucket).unwrap();
                self.meta_bucket().remove(&meta_key(&bucket)).unwrap();
            }
        }
    }
}
