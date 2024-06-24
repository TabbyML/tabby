use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tabby_common::languages::get_language_by_ext;

fn get_git_hash(path: &Path) -> Result<String> {
    Ok(git2::Oid::hash_file(git2::ObjectType::Blob, path)?.to_string())
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

pub fn source_file_key_from_path(path: &Path) -> Option<String> {
    SourceFileKey::try_from(path)
        .map(|key| key.to_string())
        .ok()
}

pub fn check_source_file_key_matched(item_key: &str) -> bool {
    let Ok(key) = item_key.parse::<SourceFileKey>() else {
        return false;
    };

    let Ok(file_key) = SourceFileKey::try_from(key.path.as_path()) else {
        return false;
    };

    // If key doesn't match, means file has been removed / modified.
    file_key.to_string() == item_key
}
