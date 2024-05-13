use std::{
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
};

use anyhow::{bail, Context, Result};
use kv::{Batch, Bucket, Config, Json, Store};
use serde::{Deserialize, Serialize};
use tabby_common::{config::RepositoryConfig, languages::get_language_by_ext};
use tracing::info;

use crate::code::{CodeIntelligence, SourceFile};

const SOURCE_FILE_BUCKET_KEY: &str = "source_files";
const INDEX_BUCKET_KEY: &str = "indexed_files";

fn cmd_stdout(cmd: &str, args: &[&str]) -> Result<String> {
    Ok(
        String::from_utf8(Command::new(cmd).args(args).output()?.stdout)?
            .trim()
            .to_string(),
    )
}

fn get_git_hash(path: &Path) -> Result<String> {
    cmd_stdout("git", &["hash-object", &path.display().to_string()])
}

#[derive(Deserialize, Serialize, Debug)]
struct SourceFileKey {
    path: PathBuf,
    language: String,
    git_hash: String,
}

impl FromStr for SourceFileKey {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

impl TryFrom<&Path> for SourceFileKey {
    type Error = anyhow::Error;

    fn try_from(path: &Path) -> Result<Self> {
        if !path.is_file() {
            bail!("Path is not a file");
        }

        let git_hash = get_git_hash(path)?;
        let ext = path.extension().context("Failed to get extension")?;
        let Some(lang) = get_language_by_ext(ext) else {
            bail!("Unknown language for extension {:?}", ext);
        };
        Ok(Self {
            path: path.to_owned(),
            language: lang.language().to_string(),
            git_hash: git_hash.to_string(),
        })
    }
}

impl ToString for SourceFileKey {
    fn to_string(&self) -> String {
        serde_json::to_string(&self).expect("Failed to serialize SourceFileKey")
    }
}

pub struct CacheStore {
    store: Store,
    code: CodeIntelligence,
}

impl CacheStore {
    pub fn new(path: PathBuf) -> Self {
        Self {
            store: Store::new(Config::new(path)).expect("Failed to create repository store"),
            code: CodeIntelligence::default(),
        }
    }

    fn index_bucket(&self) -> Bucket<String, String> {
        self.store
            .bucket(Some(INDEX_BUCKET_KEY))
            .expect("Failed to access indexed files bucket")
    }

    pub fn check_indexed(&self, path: &Path) -> (String, bool) {
        let key = SourceFileKey::try_from(path)
            .expect("Failed to create source file key")
            .to_string();
        let indexed = self
            .index_bucket()
            .contains(&key)
            .expect("Failed to read index bucket");
        (key, indexed)
    }

    pub fn apply_indexed(&self, batch: Batch<String, String>) {
        self.index_bucket()
            .batch(batch)
            .expect("Failed to commit batched index update")
    }

    #[must_use]
    pub fn prepare_garbage_collection_for_indexed_files(
        &self,
        key_remover: impl Fn(&String),
    ) -> impl FnOnce() + '_ {
        info!("Started cleaning up 'indexed_files' bucket");
        let bucket = self.index_bucket();
        let mut batch = Batch::new();

        let mut num_keep = 0;
        let mut num_removed = 0;

        bucket
            .iter()
            .filter_map(|item| {
                let item = item.expect("Failed to read item");
                let item_key: String = item.key().expect("Failed to get key");
                if is_item_key_matched(&item_key) {
                    num_keep += 1;
                    None
                } else {
                    num_removed += 1;
                    Some(item_key)
                }
            })
            .inspect(key_remover)
            .for_each(|key| batch.remove(&key).expect("Failed to remove key"));

        info!("Finished garbage collection for 'indexed_files': {num_keep} items kept, {num_removed} items removed");
        move || {
            bucket
                .batch(batch)
                .expect("Failed to execute batched delete")
        }
    }

    pub fn get_source_file(
        &mut self,
        config: &RepositoryConfig,
        path: &Path,
    ) -> Option<SourceFile> {
        let key: String = SourceFileKey::try_from(path).ok()?.to_string();

        let dataset_bucket: Bucket<String, Json<Option<SourceFile>>> = self
            .store
            .bucket(Some(SOURCE_FILE_BUCKET_KEY))
            .expect("Could not access dataset bucket");

        if let Some(source_file) = dataset_bucket
            .get(&key)
            .expect("Failed to read key from dataset bucket")
            .map(|Json(file)| file)
        {
            source_file
        } else {
            let source_file = self.code.create_source_file(config, path);
            let json = Json(source_file);
            dataset_bucket
                .set(&key, &json)
                .expect("Failed to write source file to dataset bucket");
            json.0
        }
    }

    pub fn garbage_collection_for_source_files(&self) {
        info!("Started cleaning up 'source_files' bucket");
        let bucket: Bucket<String, Json<SourceFile>> = self
            .store
            .bucket(Some(SOURCE_FILE_BUCKET_KEY))
            .expect("Could not access dataset bucket");

        let mut batch = Batch::new();
        let mut num_keep = 0;
        let mut num_removed = 0;

        bucket
            .iter()
            .filter_map(|item| {
                let item = item.expect("Failed to read item");
                let item_key: String = item.key().expect("Failed to get key");
                if is_item_key_matched(&item_key) {
                    num_keep += 1;
                    None
                } else {
                    num_removed += 1;
                    Some(item_key)
                }
            })
            .for_each(|key| batch.remove(&key).expect("Failed to remove key"));

        info!(
            "Finished garbage collection for 'source_files': {} items kept, {} items removed",
            num_keep, num_removed
        );
        bucket.batch(batch).expect("to batch remove staled files");
    }
}

fn is_item_key_matched(item_key: &str) -> bool {
    let Ok(key) = item_key.parse::<SourceFileKey>() else {
        return false;
    };

    let Ok(file_key) = SourceFileKey::try_from(key.path.as_path()) else {
        return false;
    };

    // If key doesn't match, means file has been removed / modified.
    file_key.to_string() == item_key
}
