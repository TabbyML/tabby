//! Common tabby types and utilities.
//! Defines common types and utilities used across multiple tabby subprojects, especially serialization and deserialization targets.
pub mod api;
pub mod config;
pub mod constants;
pub mod index;
pub mod languages;
pub mod path;
pub mod registry;
pub mod terminal;
pub mod usage;

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fs::File,
    future::Future,
    io::{BufReader, Error},
    marker::PhantomData,
    ops::Range,
    path::PathBuf,
};

use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use path::dataset_dir;
use serde::{Deserialize, Serialize};
use serde_jsonlines::JsonLinesReader;
use tokio::sync::RwLock;

#[derive(Serialize, Deserialize)]
pub struct SourceFile {
    pub git_url: String,
    pub filepath: String,
    pub content: String,
    pub language: String,
    pub max_line_length: usize,
    pub avg_line_length: f32,
    pub alphanum_fraction: f32,
    pub tags: Vec<Tag>,
}

impl SourceFile {
    pub fn files_jsonl() -> PathBuf {
        dataset_dir().join("files.jsonl")
    }

    pub fn all() -> Result<impl Iterator<Item = Self>, Error> {
        let files = glob::glob(format!("{}*", Self::files_jsonl().display()).as_str()).unwrap();
        let iter = files.filter_map(|x| x.ok()).flat_map(|path| {
            let fp = BufReader::new(File::open(path).unwrap());
            let reader = JsonLinesReader::new(fp);
            reader.read_all::<SourceFile>().filter_map(|x| x.ok())
        });
        Ok(iter)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Point {
    pub row: usize,
    pub column: usize,
}

impl Point {
    pub fn new(row: usize, column: usize) -> Self {
        Self { row, column }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Tag {
    pub range: Range<usize>,
    pub name_range: Range<usize>,
    pub utf16_column_range: Range<usize>,
    pub span: Range<Point>,
    pub line_range: Range<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs: Option<String>,
    pub is_definition: bool,
    pub syntax_type_name: String,
}

#[derive(Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct Package {
    pub language: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

#[derive(Default, Serialize, Deserialize)]
pub struct DependencyFile {
    pub direct: Vec<Package>,
}

pub struct Cache<T> {
    value: RwLock<Option<T>>,
}

impl<T> Cache<T> {
    pub async fn new() -> Self {
        Cache {
            value: Default::default(),
        }
    }

    pub async fn invalidate(&self) {
        *self.value.write().await = None;
    }

    pub async fn get_or_refresh<F, E>(&self, refresh: impl Fn() -> F) -> Result<T, E>
    where
        T: Clone,
        F: Future<Output = Result<T, E>>,
    {
        let value = self.value.read().await;
        if let Some(value) = &*value {
            Ok(value.clone())
        } else {
            drop(value);
            let mut value = self.value.write().await;
            let generated = refresh().await?;
            *value = Some(generated.clone());
            Ok(generated)
        }
    }

    pub async fn update(&self, f: impl FnOnce(&mut T)) {
        let mut lock = self.value.write().await;
        lock.as_mut().map(f);
    }

    pub async fn set(&self, value: T) {
        *self.value.write().await = Some(value);
    }
}
