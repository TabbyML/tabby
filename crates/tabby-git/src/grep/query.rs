use anyhow::bail;
use grep::{
    regex::RegexMatcher,
    searcher::{BinaryDetection, SearcherBuilder},
};
use ignore::types::TypesBuilder;

use super::searcher::GrepSearcher;

#[derive(Default, Clone)]
pub struct GrepQuery {
    patterns: Vec<String>,
    negative_patterns: Vec<String>,

    file_patterns: Vec<String>,
    negative_file_patterns: Vec<String>,

    file_types: Vec<String>,
    negative_file_types: Vec<String>,
}

impl GrepQuery {
    pub fn builder() -> GrepQueryBuilder {
        GrepQueryBuilder::default()
    }

    pub fn searcher(&self) -> anyhow::Result<GrepSearcher> {
        let pattern_matcher = if self.patterns.is_empty() {
            None
        } else {
            Some(RegexMatcher::new_line_matcher(&self.patterns.join("|"))?)
        };

        let negative_pattern_matcher = if self.negative_patterns.is_empty() {
            None
        } else {
            Some(RegexMatcher::new_line_matcher(
                &self.negative_patterns.join("|"),
            )?)
        };

        let file_pattern_matcher = if self.file_patterns.is_empty() {
            None
        } else {
            Some(RegexMatcher::new_line_matcher(
                &self.file_patterns.join("|"),
            )?)
        };

        let negative_file_pattern_matcher = if self.negative_file_patterns.is_empty() {
            None
        } else {
            Some(RegexMatcher::new_line_matcher(
                &self.negative_file_patterns.join("|"),
            )?)
        };

        if pattern_matcher.is_none()
            && negative_pattern_matcher.is_none()
            && file_pattern_matcher.is_none()
            && negative_file_pattern_matcher.is_none()
        {
            bail!("No patterns specified")
        }

        let file_type_matcher = if self.file_types.is_empty() && self.negative_file_types.is_empty()
        {
            None
        } else {
            let mut types_builder = TypesBuilder::new();
            types_builder.add_defaults();
            for file_type in &self.file_types {
                types_builder.select(file_type);
            }
            for file_type in &self.negative_file_types {
                types_builder.negate(file_type);
            }

            Some(types_builder.build()?)
        };

        let searcher = SearcherBuilder::new()
            .binary_detection(BinaryDetection::quit(b'\x00'))
            .before_context(3)
            .line_number(true)
            .after_context(3)
            .build();

        Ok(GrepSearcher::new(
            !self.file_patterns.is_empty() || !self.file_types.is_empty(),
            !self.patterns.is_empty(),
            pattern_matcher,
            negative_pattern_matcher,
            file_pattern_matcher,
            negative_file_pattern_matcher,
            file_type_matcher,
            searcher,
        ))
    }
}

#[derive(Default)]
pub struct GrepQueryBuilder {
    query: GrepQuery,
}

impl GrepQueryBuilder {
    pub fn pattern<T: Into<String>>(mut self, pattern: T) -> Self {
        self.query.patterns.push(pattern.into());
        self
    }

    pub fn negative_pattern<T: Into<String>>(mut self, pattern: T) -> Self {
        self.query.negative_patterns.push(pattern.into());
        self
    }

    pub fn file_pattern<T: Into<String>>(mut self, pattern: T) -> Self {
        self.query.file_patterns.push(pattern.into());
        self
    }

    pub fn negative_file_pattern<T: Into<String>>(mut self, pattern: T) -> Self {
        self.query.negative_file_patterns.push(pattern.into());
        self
    }

    pub fn file_type<T: Into<String>>(mut self, file_type: T) -> Self {
        self.query.file_types.push(file_type.into());
        self
    }

    pub fn negative_file_type<T: Into<String>>(mut self, file_type: T) -> Self {
        self.query.negative_file_types.push(file_type.into());
        self
    }

    pub fn build(self) -> GrepQuery {
        self.query
    }
}
