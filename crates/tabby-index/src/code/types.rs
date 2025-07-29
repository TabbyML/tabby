use std::{
    ops::Range,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::indexer::{IndexId, ToIndexId};

#[derive(Serialize, Deserialize, Clone)]
pub struct SourceCode {
    pub source_file_id: String,
    pub source_id: String,
    pub git_url: String,
    pub commit: String,
    pub basedir: String,
    pub filepath: String,
    pub language: String,
    pub max_line_length: usize,
    pub avg_line_length: f32,
    pub alphanum_fraction: f32,
    pub number_fraction: f32,
    pub num_lines: usize,
    pub tags: Vec<Tag>,
}

impl ToIndexId for SourceCode {
    fn to_index_id(&self) -> IndexId {
        Self::to_index_id(&self.source_id, &self.source_file_id)
    }
}

impl SourceCode {
    pub fn read_content(&self) -> std::io::Result<String> {
        let path = self.absolute_path();
        std::fs::read_to_string(path)
    }

    pub fn absolute_path(&self) -> PathBuf {
        Path::new(&self.basedir).join(&self.filepath)
    }

    pub fn source_file_id_from_id(id: &str) -> Option<&str> {
        id.split(":::").nth(1)
    }

    pub fn to_index_id(source_id: &str, source_file_id: &str) -> IndexId {
        IndexId {
            source_id: source_id.to_owned(),
            // Source file id might be duplicated across different source_ids, we prefix it with
            // source_id to make it unique within corpus.
            id: format!("{source_id}:::{source_file_id}"),
        }
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
