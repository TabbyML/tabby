use std::path::Path;

use grep::{matcher::Matcher, regex::RegexMatcher};
use ignore::types::Types;

use super::output::GrepOutput;

pub struct GrepSearcher {
    pub require_file_match: bool,
    pub require_content_match: bool,

    pattern_matcher: Option<RegexMatcher>,
    negative_pattern_matcher: Option<RegexMatcher>,

    file_pattern_matcher: Vec<RegexMatcher>,
    negative_file_pattern_matcher: Option<RegexMatcher>,

    file_type_matcher: Option<Types>,
    searcher: grep::searcher::Searcher,
}

pub enum GrepFileMatch {
    NoPattern,
    Matched,
    NotMatched,
}

impl GrepSearcher {
    pub fn new(
        require_file_match: bool,
        require_content_match: bool,
        pattern_matcher: Option<RegexMatcher>,
        negative_pattern_matcher: Option<RegexMatcher>,
        file_pattern_matcher: Vec<RegexMatcher>,
        negative_file_pattern_matcher: Option<RegexMatcher>,
        file_type_matcher: Option<Types>,
        searcher: grep::searcher::Searcher,
    ) -> Self {
        Self {
            require_file_match,
            require_content_match,
            pattern_matcher,
            negative_pattern_matcher,
            file_pattern_matcher,
            negative_file_pattern_matcher,
            file_type_matcher,
            searcher,
        }
    }

    fn file_matched(&self, path: &Path) -> anyhow::Result<GrepFileMatch> {
        let path_bytes = path.display().to_string().into_bytes();
        if let Some(ref matcher) = self.negative_file_pattern_matcher {
            if matcher.is_match(&path_bytes)? {
                return Ok(GrepFileMatch::NotMatched);
            }
        }

        let mut matched = GrepFileMatch::NoPattern;
        if let Some(ref file_type_matcher) = self.file_type_matcher {
            match file_type_matcher.matched(path, false) {
                ignore::Match::None => {
                    // Do nothing.
                }
                ignore::Match::Ignore(_) => {
                    return Ok(GrepFileMatch::NotMatched);
                }
                ignore::Match::Whitelist(_glob) => {
                    matched = GrepFileMatch::Matched;
                }
            };
        };

        for matcher in &self.file_pattern_matcher {
            if matcher.is_match(&path_bytes)? {
                matched = GrepFileMatch::Matched;
            } else {
                matched = GrepFileMatch::NotMatched;
                break;
            }
        }

        Ok(matched)
    }

    pub fn search(&mut self, content: &[u8], output: &mut GrepOutput) -> anyhow::Result<()> {
        let file_matched = self.file_matched(output.path())?;
        if let GrepFileMatch::NotMatched = file_matched {
            output.file_negated = true;
            return Ok(());
        }

        if let GrepFileMatch::Matched = file_matched {
            output.file_matched = true;
        }

        if let Some(ref matcher) = self.pattern_matcher {
            self.searcher
                .search_reader(matcher, content, output.sink(matcher))?;
        };

        if let Some(ref matcher) = self.negative_pattern_matcher {
            self.searcher
                .search_reader(matcher, content, output.negative_sink())?;
        }

        Ok(())
    }
}
