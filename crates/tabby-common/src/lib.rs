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
    fs::File,
    io::BufReader,
    ops::Range,
    path::{Path, PathBuf},
};

use path::dataset_dir;
use serde::{Deserialize, Serialize};
use serde_jsonlines::JsonLinesReader;

#[derive(Serialize, Deserialize, Clone)]
pub struct SourceFile {
    pub git_url: String,
    pub basedir: String,
    pub filepath: String,
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

    pub fn all() -> impl Iterator<Item = Self> {
        let files = glob::glob(format!("{}*", Self::files_jsonl().display()).as_str()).unwrap();

        files.filter_map(|x| x.ok()).flat_map(|path| {
            let fp = BufReader::new(File::open(path).unwrap());
            let reader = JsonLinesReader::new(fp);
            reader.read_all::<SourceFile>().filter_map(|x| x.ok())
        })
    }

    pub fn read_content(&self) -> std::io::Result<String> {
        let path = Path::new(&self.basedir).join(&self.filepath);
        std::fs::read_to_string(path)
    }

    pub fn read_file_size(&self) -> usize {
        let path = Path::new(&self.basedir).join(&self.filepath);
        std::fs::metadata(path).map(|x| x.len()).unwrap_or_default() as usize
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

#[derive(Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
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
