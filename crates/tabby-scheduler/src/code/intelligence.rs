use std::{fs::read_to_string, path::Path};

use tabby_common::{config::RepositoryConfig, languages::get_language_by_ext};
use text_splitter::{Characters, TextSplitter};
use tracing::warn;
use tree_sitter_tags::TagsContext;

use super::languages;
pub use super::types::{Point, SourceCode, Tag};

pub struct CodeIntelligence {
    context: TagsContext,
    splitter: TextSplitter<Characters>,
}

impl Default for CodeIntelligence {
    fn default() -> Self {
        Self {
            context: TagsContext::new(),
            splitter: TextSplitter::default().with_trim_chunks(true),
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

    // FIXME(meng): implement with treesitter based CodeSplitter.
    pub fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = &'text str> + 'splitter {
        self.splitter.chunks(text, 192)
    }
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
