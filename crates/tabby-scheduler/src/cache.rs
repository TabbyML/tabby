use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
};

use anyhow::{bail, Context, Result};
use kv::{Batch, Bucket, Config, Json, Store};
use serde::{Deserialize, Serialize};
use tabby_common::{config::RepositoryConfig, languages::get_language_by_ext, SourceFile};
use tracing::{info, warn};

use crate::code::CodeIntelligence;

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
pub(crate) struct SourceFileKey {
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

    pub fn is_indexed(&self, key: &SourceFileKey) -> bool {
        self.index_bucket()
            .contains(&key.to_string())
            .expect("Failed to read index bucket")
    }

    pub fn set_indexed(&self, key: &SourceFileKey) {
        self.index_bucket()
            .set(&key.to_string(), &String::new())
            .expect("Failed to write to index bucket");
    }

    pub fn cleanup_old_indexed_files(&self, key_remover: impl Fn(&String)) {
        info!("Cleaning up indexed file cache");
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
            .for_each(|key| {
                batch
                    .remove(&key)
                    .expect("Failed to remove indexed source file")
            });

        info!("Finished cleaning up indexed files: {num_keep} items kept, {num_removed} items removed");
        bucket
            .batch(batch)
            .expect("Failed to execute batched delete");
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
            let source_file = create_source_file(config, path, &mut self.code);
            let json = Json(source_file);
            dataset_bucket
                .set(&key, &json)
                .expect("Failed to write source file to dataset bucket");
            json.0
        }
    }

    pub fn cleanup_old_source_files(&self) {
        info!("Cleaning up synced file cache");
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
            "Finished garbage collection: {} items kept, {} items removed",
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
        warn!("Unknown language for extension {:?}", ext);
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
