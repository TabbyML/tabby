use std::str::FromStr;

use anyhow::bail;
use grep::{
    regex::{RegexMatcher, RegexMatcherBuilder},
    searcher::{BinaryDetection, SearcherBuilder},
};
use ignore::types::TypesBuilder;

use super::searcher::GrepSearcher;

#[derive(Default, Clone, Debug)]
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
            let pattern = self.patterns.join("|");
            let case_insensitive = !has_uppercase_literal(&pattern);
            Some(
                RegexMatcherBuilder::new()
                    .case_insensitive(case_insensitive)
                    .line_terminator(Some(b'\n'))
                    .build(&self.patterns.join("|"))?,
            )
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
            let mut matchers = vec![];
            for p in &self.file_patterns {
                let case_insensitive = !has_uppercase_literal(p);
                let matcher = RegexMatcherBuilder::new()
                    .case_insensitive(case_insensitive)
                    .line_terminator(Some(b'\n'))
                    .build(p)?;
                matchers.push(matcher);
            }
            matchers
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
            && self.file_types.is_empty()
            && self.negative_file_types.is_empty()
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
        for (negative, part) in tokenize_query(s) {
            if negative {
                match part {
                    _ if part.starts_with("lang:") => {
                        builder = builder.negative_file_type(&part[5..])
                    }
                    _ if part.starts_with("f:") => {
                        builder = builder.negative_file_pattern(&part[2..])
                    }
                    _ => builder = builder.negative_pattern(part),
                }
            } else {
                match part {
                    _ if part.starts_with("lang:") => builder = builder.file_type(&part[5..]),
                    _ if part.starts_with("f:") => builder = builder.file_pattern(&part[2..]),
                    _ => builder = builder.pattern(part),
                }
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

fn has_uppercase_literal(expr: &str) -> bool {
    expr.chars().any(|c| c.is_ascii_uppercase())
}

/// Tokenize a query string, and respectes quoted strings.
/// When a token is prefixed with a `-`, it is considered a negative pattern.
///
/// Quote characters can be escaped with a backslash.
/// Returns the list of tokens, and whether they are negative patterns.
fn tokenize_query(query: &str) -> Vec<(bool, String)> {
    let mut tokens = vec![];
    let mut current = String::new();
    let mut negative = false;
    let mut quoted = false;
    let mut escaped = false;

    for c in query.chars() {
        if escaped {
            current.push(c);
            escaped = false;
            continue;
        }

        match c {
            ' ' if !quoted => {
                if !current.is_empty() {
                    tokens.push((negative, current.clone()));
                    current.clear();
                    negative = false;
                }
            }
            '-' if !quoted => {
                if !current.is_empty() {
                    tokens.push((negative, current.clone()));
                    current.clear();
                }
                negative = true;
            }
            '"' => {
                if quoted {
                    tokens.push((negative, current.clone()));
                    current.clear();
                }
                quoted = !quoted;
            }
            '\\' => {
                escaped = true;
            }
            _ => {
                current.push(c);
            }
        }
    }

    if !current.is_empty() {
        tokens.push((negative, current));
    }

    tokens
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

    #[test]
    fn test_tokenize_query() {
        let query = r#"lang:rust -f:*.rs foo bar -baz "qux quux", -"corge grault" "\"abc\" dd""#;
        let tokens = tokenize_query(query);
        assert_eq!(
            tokens,
            vec![
                (false, "lang:rust".to_owned()),
                (true, "f:*.rs".to_owned()),
                (false, "foo".to_owned()),
                (false, "bar".to_owned()),
                (true, "baz".to_owned()),
                (false, "qux quux".to_owned()),
                (false, ",".to_owned()),
                (true, "corge grault".to_owned()),
                (true, "\"abc\" dd".to_owned())
            ]
        );
    }
}
