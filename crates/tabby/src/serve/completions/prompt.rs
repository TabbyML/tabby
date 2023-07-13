use strfmt::strfmt;
use tabby_common::path::index_dir;
use tantivy::{Index, ReloadPolicy, IndexReader, Searcher};
use tracing::warn;

use super::Segments;

pub struct PromptBuilder {
    prompt_template: Option<String>,
    reader: Option<IndexReader>
}

impl PromptBuilder {
    pub fn new(prompt_template: Option<String>) -> Self {
        let index = Index::open_in_dir(index_dir());
        let reader = index.and_then(|index| index.reader_builder().reload_policy(ReloadPolicy::OnCommit).try_into());

        if let Err(err) = &reader {
            warn!("Failed to open index in {:?}: {:?}", index_dir(), err);
        }

        PromptBuilder {
            prompt_template,
            reader: reader.ok(),
        }
    }

    fn build_prompt(&self, prefix: String, suffix: String) -> String {
        if let Some(prompt_template) = &self.prompt_template {
            strfmt!(prompt_template, prefix => prefix, suffix => suffix).unwrap()
        } else {
            prefix
        }
    }

    pub fn build(&self, language: &str, segments: Segments) -> String {
        let segments = self.rewrite(language, segments);
        if let Some(suffix) = segments.suffix {
            self.build_prompt(segments.prefix, suffix)
        } else {
            self.build_prompt(segments.prefix, "".to_owned())
        }
    }

    fn rewrite(&self, language: &str, segments: Segments) -> Segments {
        if let Some(reader) = &self.reader {
            self.rewrite_with_index(reader.searcher(), language, segments)
        } else {
            segments
        }
    }

    fn rewrite_with_index(&self, searcher: Searcher, language: &str, segments: Segments) -> Segments {
        // FIXME(meng): implement this.
        segments
    }
}
