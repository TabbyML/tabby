pub mod config;
pub mod events;
pub mod index;
pub mod languages;
pub mod path;
pub mod usage;

use std::{
    fs::File,
    io::{BufReader, Error},
    ops::Range,
};

use path::dataset_dir;
use serde::{Deserialize, Serialize};
use serde_jsonlines::JsonLinesReader;

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
    pub fn all() -> Result<impl Iterator<Item = Self>, Error> {
        let iter = dataset_dir().read_dir()?.flat_map(|path| {
            let path = path.unwrap().path();
            let fp = BufReader::new(File::open(path).unwrap());
            let reader = JsonLinesReader::new(fp);
            reader.read_all::<SourceFile>().map(|x| x.unwrap())
        });
        Ok(iter)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Tag {
    pub range: Range<usize>,
    pub name_range: Range<usize>,
    pub line_range: Range<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs: Option<String>,
    pub is_definition: bool,
    pub syntax_type_name: String,
}
