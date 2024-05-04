use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Result;

use kv::{Batch, Bucket, Config, Json, Store};

use tabby_common::{config::RepositoryConfig, languages::get_language_by_ext, SourceFile};
use tracing::{debug, warn};

use crate::code::CodeIntelligence;

const DATASET_BUCKET_PREFIX: &str = "dataset";

fn get_git_hash(path: &Path) -> Result<String> {
    let path = path.display().to_string();
    let output = Command::new("git").args(["hash-files", &path]).output()?;
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn dataset_bucket_key(git_url: impl AsRef<str>) -> String {
    format!("{DATASET_BUCKET_PREFIX}:{}", git_url.as_ref())
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

    pub fn get_or_create_source_file(
        &mut self,
        config: &RepositoryConfig,
        path: &Path,
    ) -> Option<SourceFile> {
        let git_hash = get_git_hash(path).expect("Failed to get git hash");
        let dataset_bucket: Bucket<String, Json<SourceFile>> = self
            .store
            .bucket(Some(&dataset_bucket_key(config.canonical_git_url())))
            .expect("Could not access dataset bucket");

        if let Some(source_file) = dataset_bucket
            .get(&path.display().to_string())
            .expect("Failed to read key from dataset bucket")
            .map(|Json(file)| file)
            .filter(|file| file.git_hash == git_hash)
        {
            Some(source_file)
        } else {
            if let Some(source_file) = create_source_file(config, path, &mut self.code) {
                let json = Json(source_file);
                dataset_bucket
                    .set(&json.0.filepath, &json)
                    .expect("Failed to write source file to dataset bucket");
                Some(json.0)
            } else {
                None
            }
        }
    }

    pub fn garbage_collection(&self) {
        self.store
            .buckets()
            .into_iter()
            .filter_map(|name| {
                if name.starts_with(DATASET_BUCKET_PREFIX) {
                    self.store
                        .bucket(Some(&name))
                        .ok()
                        .map(|bucket| (name, bucket))
                } else {
                    None
                }
            })
            .map(|(name, bucket)| {
                self.remove_staled_files(&bucket);
                name
            })
            .for_each(|name| {
                self.store
                    .drop_bucket(name)
                    .expect("Failed to remove bucket")
            });
    }

    fn remove_staled_files(&self, bucket: &Bucket<String, Json<SourceFile>>) {
        let mut batch = Batch::new();

        bucket
            .iter()
            .filter_map(|item| {
                let item = item.ok()?;
                let key: String = item.key().ok()?;
                let Json(file) = item.value().ok()?;
                let filepath = PathBuf::from(file.basedir).join(file.filepath);
                let git_hash = get_git_hash(filepath.as_path()).ok()?;
                if filepath.exists() && git_hash == file.git_hash {
                    Some(key)
                } else {
                    None
                }
            })
            .for_each(|key| batch.remove(&key).expect("Failed to remove key"));

        bucket.batch(batch).expect("to batch remove staled files");
    }
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
    let git_hash = get_git_hash(path).ok()?;
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
        git_hash,
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
