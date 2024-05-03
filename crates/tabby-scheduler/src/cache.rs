use std::{
    collections::HashSet,
    fs::read_to_string,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Result};
use ignore::Walk;
use kv::{Bucket, Config, Json, Store, Transaction, TransactionError};
use serde::{Deserialize, Serialize};
use tabby_common::{config::RepositoryConfig, languages::get_language_by_ext, SourceFile};
use tracing::{debug, warn};

use crate::code::CodeIntelligence;

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

pub struct CacheStore {
    store: Store,
}

#[derive(Serialize, Deserialize, Default)]
struct Meta {
    last_sync_commit: Option<String>,
}

impl CacheStore {
    pub fn new(path: PathBuf) -> Self {
        Self {
            store: Store::new(Config::new(path)).expect("Failed to create repository store"),
        }
    }

    fn meta_bucket(&self) -> Bucket<String, Json<Meta>> {
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
        transaction: &Transaction<String, Json<Meta>>,
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
        transaction: &Transaction<String, Json<Meta>>,
        repository: &RepositoryConfig,
    ) -> Meta {
        transaction
            .get(&meta_key(repository.canonical_git_url()))
            .expect("Failed to access repository meta")
            .map(|Json(meta)| meta)
            .unwrap_or_default()
    }

    pub fn update_source_files(&self, repositories: &[RepositoryConfig]) {
        for repository in repositories {
            debug!(
                "Refreshing source files for {}",
                repository.canonical_git_url()
            );
            self.refresh_source_files(repository);
        }
        self.retain_from(repositories);
    }

    fn refresh_source_files(&self, repository: &RepositoryConfig) {
        let dir = repository.dir();

        self.meta_bucket()
            .transaction2(
                &self.dataset_bucket(repository),
                |meta_bucket, repo_bucket| {
                    let last_sync_commit = self.get_meta(&meta_bucket, repository).last_sync_commit;
                    let current_version = get_git_commit(&dir).unwrap();

                    let Some(old_version) = last_sync_commit else {
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
                                .set(&file, &Json(source_file))
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
        meta_bucket: &Transaction<String, Json<Meta>>,
        repo_bucket: &Transaction<String, Json<SourceFile>>,
        current_version: String,
        repository: &RepositoryConfig,
    ) {
        for file in build_repository_dataset(repository) {
            repo_bucket
                .set(&file.filepath.clone(), &Json(file))
                .expect("Failed to update source file");
        }
        self.set_last_sync_commit(meta_bucket, repository, current_version);
    }

    pub fn source_files(&self) -> impl Iterator<Item = SourceFile> + '_ {
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
                debug!("Dropping bucket: {}", bucket);
                self.store.drop_bucket(&bucket).unwrap();
                self.meta_bucket().remove(&meta_key(&bucket)).unwrap();
            }
        }
    }
}

fn build_repository_dataset(
    repository: &RepositoryConfig,
) -> impl Iterator<Item = SourceFile> + '_ {
    let basedir = repository.dir();
    let walk_dir = Walk::new(basedir.as_path()).filter_map(Result::ok);

    let mut code = CodeIntelligence::default();
    walk_dir.filter_map(move |entry| create_source_file(repository, entry.path(), &mut code))
}

fn create_source_file(
    config: &RepositoryConfig,
    path: &Path,
    code: &mut CodeIntelligence,
) -> Option<SourceFile> {
    if path.is_dir() || !path.exists() {
        return None;
    }
    let relative_path = path
        .strip_prefix(&config.dir())
        .expect("Paths always begin with the prefix");

    let Some(ext) = relative_path.extension() else {
        return None;
    };

    let Some(language_info) = get_language_by_ext(ext) else {
        debug!("Unknown language for {relative_path:?}");
        return None;
    };

    let language = language_info.language();
    let contents = match read_to_string(path) {
        Ok(x) => x,
        Err(_) => {
            warn!("Failed to read {path:?}, skipping...");
            return None;
        }
    };
    let source_file = SourceFile {
        git_url: config.canonical_git_url(),
        basedir: config.dir().display().to_string(),
        filepath: relative_path.display().to_string(),
        max_line_length: metrics::max_line_length(&contents),
        avg_line_length: metrics::avg_line_length(&contents),
        alphanum_fraction: metrics::alphanum_fraction(&contents),
        tags: code.find_tags(language, &contents),
        language: language.into(),
    };
    Some(source_file)
}

mod metrics {
    use std::cmp::max;

    pub fn max_line_length(content: &str) -> usize {
        content.lines().map(|x| x.len()).reduce(max).unwrap_or(0)
    }

    pub fn avg_line_length(content: &str) -> f32 {
        let mut total = 0;
        let mut len = 0;
        for x in content.lines() {
            len += 1;
            total += x.len();
        }

        if len > 0 {
            total as f32 / len as f32
        } else {
            0.0
        }
    }

    pub fn alphanum_fraction(content: &str) -> f32 {
        let num_alphanumn: f32 = content
            .chars()
            .map(|x| f32::from(u8::from(x.is_alphanumeric())))
            .sum();
        if !content.is_empty() {
            num_alphanumn / content.len() as f32
        } else {
            0.0
        }
    }
}
