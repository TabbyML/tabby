use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{bail, Result};
use kv::{Batch, Bucket, Config, Item, Json, Store};
use tabby_common::{config::RepositoryConfig, languages::get_language_by_ext, SourceFile};
use tracing::{debug, warn};

use crate::code::CodeIntelligence;

const SOURCE_FILE_BUCKET_KEY: &str = "source_files";

fn get_git_hash(path: &Path) -> Result<String> {
    let path = path.display().to_string();
    let output = Command::new("git").args(["hash-object", &path]).output()?;
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
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

    pub fn get_source_file(
        &mut self,
        config: &RepositoryConfig,
        path: &Path,
    ) -> Option<SourceFile> {
        let git_hash = get_git_hash(path).expect("Failed to get git hash");
        let dataset_bucket: Bucket<String, Json<Option<SourceFile>>> = self
            .store
            .bucket(Some(SOURCE_FILE_BUCKET_KEY))
            .expect("Could not access dataset bucket");

        if let Some(source_file) = dataset_bucket
            .get(&git_hash)
            .expect("Failed to read key from dataset bucket")
            .map(|Json(file)| file)
        {
            source_file
        } else {
            let source_file = create_source_file(config, path, &mut self.code);
            let json = Json(source_file);
            dataset_bucket
                .set(&git_hash, &json)
                .expect("Failed to write source file to dataset bucket");
            json.0
        }
    }

    pub fn garbage_collection(&self) {
        debug!("Running garbage collection");
        let bucket = self
            .store
            .bucket(Some(SOURCE_FILE_BUCKET_KEY))
            .expect("Could not access dataset bucket");

        let mut batch = Batch::new();
        let mut num_removed = 0;

        bucket
            .iter()
            .filter_map(|item| {
                let item = item.expect("Failed to read item");
                if verify_git_hash(&item).is_err() {
                    num_removed += 1;
                    Some(item.key().expect("Failed to get key"))
                } else {
                    None
                }
            })
            .for_each(|key| batch.remove(&key).expect("Failed to remove key"));

        debug!("Removed {} staled keys", num_removed);
        bucket.batch(batch).expect("to batch remove staled files");
    }
}

fn verify_git_hash(item: &Item<String, Json<SourceFile>>) -> Result<()> {
    let key_git_hash: String = item.key()?;
    let Json(file) = item.value()?;
    let filepath = PathBuf::from(file.basedir).join(file.filepath);
    let git_hash = get_git_hash(filepath.as_path())?;

    if git_hash != key_git_hash {
        bail!("Git hash mismatch for {}", filepath.display());
    }

    Ok(())
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
        // Ignore unknown languages
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
