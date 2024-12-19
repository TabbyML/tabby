mod id;

use std::{fs::read_to_string, path::Path};

use async_stream::stream;
use futures::Stream;
use id::SourceFileId;
use tabby_common::languages::get_language_by_ext;
use text_splitter::{CodeSplitter, TextSplitter};
use tracing::warn;
use tree_sitter_tags::TagsContext;

pub use super::types::{Point, SourceCode, Tag};
use super::{
    languages::{self},
    CodeRepository,
};

pub struct CodeIntelligence;

const CHUNK_SIZE: usize = 512;

impl CodeIntelligence {
    fn find_tags(language: &str, content: &str) -> Vec<Tag> {
        let config = languages::get(language);
        let empty = Vec::new();

        let Some(config) = config else {
            return empty;
        };

        let mut context = TagsContext::new();

        let Ok((tags, has_error)) = context.generate_tags(&config.0, content.as_bytes(), None)
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

    pub fn compute_source_file_id(path: &Path) -> Option<String> {
        SourceFileId::try_from(path).map(|key| key.to_string()).ok()
    }

    pub fn check_source_file_id_matched(item_key: &str) -> bool {
        let Ok(key) = item_key.parse::<SourceFileId>() else {
            warn!("Failed to parse key: {}", item_key);
            return false;
        };

        let Ok(file_key) = SourceFileId::try_from(key.path()) else {
            return false;
        };

        // If key doesn't match, means file has been removed / modified.
        file_key.to_string() == item_key
    }

    pub fn compute_source_file(
        config: &CodeRepository,
        commit: &str,
        path: &Path,
    ) -> Option<SourceCode> {
        let source_file_id = Self::compute_source_file_id(path)?;

        if path.is_dir() || !path.exists() {
            warn!("Path {} is not a file or does not exist", path.display());
            return None;
        }
        let relative_path = path
            .strip_prefix(config.dir())
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

        let metrics::Metrics {
            max_line_length,
            avg_line_length,
            alphanum_fraction,
            num_lines,
            number_fraction,
        } = metrics::compute_metrics(&contents);

        let source_file = SourceCode {
            source_file_id,
            source_id: config.source_id.clone(),
            git_url: config.canonical_git_url(),
            commit: commit.to_owned(),
            basedir: config.dir().display().to_string(),
            filepath: relative_path.display().to_string(),
            max_line_length,
            avg_line_length,
            alphanum_fraction,
            number_fraction,
            tags: Self::find_tags(language, &contents),
            num_lines,
            language: language.into(),
        };
        Some(source_file)
    }

    fn stream_text_chunks(text: &str) -> impl Stream<Item = (usize, String)> {
        let text = text.to_owned();
        let splitter = TextSplitter::new(CHUNK_SIZE);
        stream! {
            for (offset, chunk) in splitter.chunk_indices(&text) {
                yield (offset, chunk.to_owned());
            }
        }
    }

    fn stream_code_chunks(
        text: &str,
        language: &str,
    ) -> Option<impl Stream<Item = (usize, String)>> {
        let Some(config) = languages::get(language) else {
            return None;
        };
        let chunk_size = tabby_common::languages::get_language(language)
            .chunk_size
            .unwrap_or(CHUNK_SIZE);
        let text = text.to_owned();
        let splitter = CodeSplitter::new(config.0.language.clone(), chunk_size)
            .expect("Failed to create code splitter");
        Some(stream! {
            for (offset, chunk) in splitter.chunk_indices(&text) {
                yield (offset, chunk.to_owned());
            }
        })
    }

    pub fn chunks(text: &str, language: &str) -> impl Stream<Item = (usize, String)> {
        let text = text.to_owned();
        let language = language.to_owned();
        stream! {
            let mut last_offset = 0;
            let mut last_line_number = 1;
            if let Some(stream) = Self::stream_code_chunks(&text, &language) {
                for await (offset, chunk) in stream {
                    last_line_number = line_number_from_byte_offset(&text, last_offset, last_line_number, offset);
                    last_offset = offset;
                    yield (last_line_number, chunk);
                }
            } else {
                for await (offset, chunk) in Self::stream_text_chunks(&text) {
                    last_line_number = line_number_from_byte_offset(&text, last_offset, last_line_number, offset);
                    last_offset = offset;
                    yield (last_line_number, chunk);
                }
            }
        }
    }
}

fn line_number_from_byte_offset(
    s: &str,
    last_offset: usize,
    last_line_number: usize,
    byte_offset: usize,
) -> usize {
    let mut line_number = last_line_number; // Start counting from line 1
    let mut current_offset = last_offset;

    for c in s[last_offset..].chars() {
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

    pub struct Metrics {
        pub max_line_length: usize,
        pub avg_line_length: f32,
        pub alphanum_fraction: f32,
        pub number_fraction: f32,
        pub num_lines: usize,
    }

    pub fn compute_metrics(content: &str) -> Metrics {
        let mut metrics = Metrics {
            max_line_length: 0,
            avg_line_length: 0.0,
            alphanum_fraction: 0.0,
            number_fraction: 0.0,
            num_lines: 0,
        };
        // Compute metrics in single loop.
        for x in content.lines() {
            metrics.num_lines += 1;
            let line_length = x.len();
            metrics.max_line_length = max(metrics.max_line_length, line_length);
            metrics.avg_line_length += line_length as f32;
            for c in x.chars() {
                if c.is_alphanumeric() {
                    metrics.alphanum_fraction += 1.0;
                }
                if c.is_numeric() {
                    metrics.number_fraction += 1.0;
                }
            }
        }

        metrics.avg_line_length /= metrics.num_lines as f32;
        metrics.alphanum_fraction /= content.len() as f32;
        metrics.number_fraction /= content.len() as f32;

        metrics
    }
}

#[cfg(test)]
mod tests {
    use serial_test::file_serial;
    use tabby_common::path::set_tabby_root;
    use tracing_test::traced_test;

    use super::*;
    use crate::testutils::{get_repository_config, get_rust_source_file, get_tabby_root};

    #[test]
    #[traced_test]
    #[file_serial(set_tabby_root)]
    fn test_create_source_file() {
        set_tabby_root(get_tabby_root());
        let config = get_repository_config();
        let source_file =
            CodeIntelligence::compute_source_file(&config, "commit", &get_rust_source_file())
                .expect("Failed to create source file");

        // check source_file properties
        assert_eq!(source_file.language, "rust");
        assert_eq!(source_file.tags.len(), 3);
        assert_eq!(source_file.filepath, "rust.rs");
        assert_eq!(source_file.commit, "commit");
    }
}
