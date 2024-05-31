use std::{collections::HashMap, fs::read_to_string, path::Path};

use tabby_common::{config::RepositoryConfig, languages::get_language_by_ext};
use text_splitter::{Characters, ExperimentalCodeSplitter, TextSplitter};
use tracing::warn;
use tree_sitter_tags::TagsContext;

use super::languages;
pub use super::types::{Point, SourceCode, Tag};

pub struct CodeIntelligence {
    context: TagsContext,
    splitter: TextSplitter<Characters>,
    code_splitters: HashMap<String, ExperimentalCodeSplitter<Characters>>,
}

const CHUNK_SIZE: usize = 256;

impl Default for CodeIntelligence {
    fn default() -> Self {
        Self {
            context: TagsContext::new(),
            splitter: TextSplitter::new(CHUNK_SIZE),
            code_splitters: super::languages::all()
                .map(|(name, config)| {
                    let name = name.to_string();
                    let splitter =
                        ExperimentalCodeSplitter::new(config.0.language.clone(), CHUNK_SIZE)
                            .expect("Failed to create code splitter");
                    (name, splitter)
                })
                .collect(),
        }
    }
}

impl CodeIntelligence {
    pub fn find_tags(&mut self, language: &str, content: &str) -> Vec<Tag> {
        let config = languages::get(language);
        let empty = Vec::new();

        let Some(config) = config else {
            return empty;
        };

        let Ok((tags, has_error)) = self
            .context
            .generate_tags(&config.0, content.as_bytes(), None)
        else {
            return empty;
        };

        if has_error {
            return empty;
        }

        tags.filter_map(|x| x.ok())
            .map(|x| Tag {
                range: x.range,
                name_range: x.name_range,
                utf16_column_range: x.utf16_column_range,
                line_range: x.line_range,
                docs: x.docs,
                is_definition: x.is_definition,
                syntax_type_name: config.0.syntax_type_name(x.syntax_type_id).to_owned(),
                span: Point::new(x.span.start.row, x.span.start.column)
                    ..Point::new(x.span.end.row, x.span.end.column),
            })
            .collect()
    }

    pub fn create_source_file(
        &mut self,
        config: &RepositoryConfig,
        path: &Path,
    ) -> Option<SourceCode> {
        if path.is_dir() || !path.exists() {
            warn!("Path {} is not a file or does not exist", path.display());
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
        let source_file = SourceCode {
            git_url: config.canonical_git_url(),
            basedir: config.dir().display().to_string(),
            filepath: relative_path.display().to_string(),
            max_line_length: metrics::max_line_length(&contents),
            avg_line_length: metrics::avg_line_length(&contents),
            alphanum_fraction: metrics::alphanum_fraction(&contents),
            tags: self.find_tags(language, &contents),
            language: language.into(),
        };
        Some(source_file)
    }

    pub fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
        language: &'text str,
    ) -> Box<dyn Iterator<Item = (usize, &'text str)> + 'splitter + Send> {
        if let Some(splitter) = self.code_splitters.get(language) {
            Box::new(
                splitter
                    .chunk_indices(text)
                    .map(|(offset, chunk)| (line_number_from_byte_offset(text, offset), chunk)),
            )
        } else {
            Box::new(
                self.splitter
                    .chunk_indices(text)
                    .map(|(offset, chunk)| (line_number_from_byte_offset(text, offset), chunk)),
            )
        }
    }
}

fn line_number_from_byte_offset(s: &str, byte_offset: usize) -> usize {
    let mut line_number = 1; // Start counting from line 1
    let mut current_offset = 0;

    for c in s.chars() {
        if c == '\n' {
            line_number += 1;
        }
        current_offset += c.len_utf8();
        if current_offset >= byte_offset {
            break;
        }
    }

    line_number
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tabby_common::path::set_tabby_root;
    use tracing_test::traced_test;

    use super::*;

    fn get_tabby_root() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("testdata");
        path
    }

    fn get_repository_config() -> RepositoryConfig {
        RepositoryConfig::new("https://github.com/TabbyML/tabby")
    }

    fn get_rust_source_file() -> PathBuf {
        let mut path = get_tabby_root();
        path.push("repositories");
        path.push("https_github.com_TabbyML_tabby");
        path.push("rust.rs");
        path
    }

    #[test]
    #[traced_test]
    fn test_create_source_file() {
        set_tabby_root(get_tabby_root());
        let config = get_repository_config();
        let mut code = CodeIntelligence::default();
        let source_file = code
            .create_source_file(&config, &get_rust_source_file())
            .expect("Failed to create source file");

        // check source_file properties
        assert_eq!(source_file.language, "rust");
        assert_eq!(source_file.tags.len(), 3);
        assert_eq!(source_file.filepath, "rust.rs");
    }
}
