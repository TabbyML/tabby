use std::str::FromStr;

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
    #[cfg(test)]
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
            vec![]
        } else {
            let mut matcher = vec![];
            for p in &self.file_patterns {
                matcher.push(RegexMatcher::new_line_matcher(p)?);
            }
            matcher
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
            && file_pattern_matcher.is_empty()
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

impl FromStr for GrepQuery {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut builder = GrepQueryBuilder::default();
        for part in s.split_whitespace() {
            if part.starts_with('-') {
                builder = match part {
                    _ if part.starts_with("-lang:") => builder.negative_file_type(&part[6..]),
                    _ if part.starts_with("-f:") => builder.negative_file_pattern(&part[3..]),
                    _ => builder.negative_pattern(&part[1..]),
                };
            } else {
                builder = match part {
                    _ if part.starts_with("lang:") => builder.file_type(&part[5..]),
                    _ if part.starts_with("f:") => builder.file_pattern(&part[2..]),
                    _ => builder.pattern(part),
                };
            }
        }

        Ok(builder.build())
    }
}

#[derive(Default)]
pub struct GrepQueryBuilder {
    query: GrepQuery,
}

impl GrepQueryBuilder {
    pub fn pattern<T: Into<String>>(mut self, pattern: T) -> Self {
        let pattern = pattern.into();
        if !pattern.is_empty() {
            self.query.patterns.push(pattern);
        }
        self
    }

    pub fn negative_pattern<T: Into<String>>(mut self, pattern: T) -> Self {
        let pattern = pattern.into();
        if !pattern.is_empty() {
            self.query.negative_patterns.push(pattern);
        }
        self
    }

    pub fn file_pattern<T: Into<String>>(mut self, pattern: T) -> Self {
        let pattern = pattern.into();
        if !pattern.is_empty() {
            self.query.file_patterns.push(pattern);
        }
        self
    }

    pub fn negative_file_pattern<T: Into<String>>(mut self, pattern: T) -> Self {
        let pattern = pattern.into();
        if !pattern.is_empty() {
            self.query.negative_file_patterns.push(pattern);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_query() {
        let query: GrepQuery = "lang:rust -f:*.rs foo bar -baz".parse().unwrap();
        assert_eq!(query.patterns, vec!["foo", "bar"]);
        assert_eq!(query.negative_patterns, vec!["baz"]);
        assert_eq!(query.negative_file_patterns, vec!["*.rs"]);
        assert_eq!(query.file_types, vec!["rust"]);
    }
}
