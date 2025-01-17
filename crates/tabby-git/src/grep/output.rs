use std::{
    io::BufRead,
    path::{Path, PathBuf},
};

use grep::{matcher::Matcher, regex::RegexMatcher, searcher::Sink};
use tracing::debug;

use super::{GrepFile, GrepLine, GrepSubMatch, GrepTextOrBase64};

pub struct GrepOutput {
    path: PathBuf,
    lines: Vec<GrepLine>,

    tx: tokio::sync::mpsc::Sender<GrepFile>,

    content_matched: bool,
    content_negated: bool,

    pub file_matched: bool,
    pub file_negated: bool,
}

impl std::fmt::Debug for GrepOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrepOutput")
            .field("content_matched", &self.content_matched)
            .field("content_negated", &self.content_negated)
            .field("file_matched", &self.file_matched)
            .field("file_negated", &self.file_negated)
            .finish()
    }
}

impl GrepOutput {
    pub fn new(path: PathBuf, tx: tokio::sync::mpsc::Sender<GrepFile>) -> Self {
        Self {
            path: path.to_owned(),
            lines: Vec::new(),
            tx,

            file_matched: false,
            file_negated: false,

            content_matched: false,
            content_negated: false,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn sink<'output, 'a>(
        &'output mut self,
        matcher: &'a RegexMatcher,
    ) -> GrepMatchSink<'output, 'a> {
        GrepMatchSink {
            output: self,
            matcher,
        }
    }

    pub fn negative_sink(&mut self) -> GrepNegativeMatchSink<'_> {
        GrepNegativeMatchSink { output: self }
    }

    fn record(&mut self, line: GrepLine) {
        self.lines.push(line);
    }

    pub fn flush(&mut self, require_file_match: bool, require_content_match: bool, content: &[u8]) {
        // If file or content is negated, we don't want to send the file.
        if self.file_negated || self.content_negated {
            return;
        }

        if require_file_match && !self.file_matched {
            return;
        }

        if require_content_match && !self.content_matched {
            return;
        }

        let lines = if self.content_matched {
            std::mem::take(&mut self.lines)
        } else {
            match read_lines(content) {
                Ok(lines) => lines,
                Err(e) => {
                    debug!("Failed to read file: {:?}", e);
                    vec![]
                }
            }
        };

        let file = GrepFile {
            path: self.path.clone(),
            lines,
        };

        match self.tx.blocking_send(file) {
            Ok(_) => {}
            Err(_) => {
                // Request got cancelled, we can safely ignore the error.
            }
        }
    }
}

fn read_lines(content: &[u8]) -> anyhow::Result<Vec<GrepLine>> {
    let reader = std::io::BufReader::new(content);
    let line_reader = reader.lines().take(5);

    let mut lines = vec![];
    let mut line_number = 1;
    let mut byte_offset = 0;
    for line in line_reader {
        let line = line? + "\n";
        let bytes_length = line.len();
        lines.push(GrepLine {
            line: GrepTextOrBase64::Text(line),
            byte_offset,
            line_number,
            sub_matches: vec![],
        });

        byte_offset += bytes_length;
        line_number += 1;
    }

    Ok(lines)
}

pub struct GrepMatchSink<'output, 'a> {
    output: &'output mut GrepOutput,
    matcher: &'a RegexMatcher,
}

impl Sink for GrepMatchSink<'_, '_> {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _searcher: &grep::searcher::Searcher,
        mat: &grep::searcher::SinkMatch<'_>,
    ) -> Result<bool, Self::Error> {
        self.output.content_matched = true;

        // 1. Search is always done in single-line mode.
        let line = mat.lines().next().expect("Have at least one line");

        // 2. Collect all matches in the line.
        let mut matches: Vec<GrepSubMatch> = vec![];
        self.matcher.find_iter(line, |m| {
            matches.push(GrepSubMatch {
                bytes_start: m.start(),
                bytes_end: m.end(),
            });
            true
        })?;

        let line = GrepTextOrBase64::Base64(line.to_owned());

        // 3. Create a GrepLine object and add it to the file.
        let line = GrepLine {
            line,
            byte_offset: mat.absolute_byte_offset() as usize,
            line_number: mat.line_number().expect("Have line number") as usize,
            sub_matches: matches,
        };

        self.output.record(line);
        Ok(true)
    }

    fn context(
        &mut self,
        _searcher: &grep::searcher::Searcher,
        context: &grep::searcher::SinkContext<'_>,
    ) -> Result<bool, Self::Error> {
        let line = context.bytes();

        let line = match std::str::from_utf8(line) {
            Ok(s) => GrepTextOrBase64::Text(s.to_owned()),
            Err(_) => GrepTextOrBase64::Base64(line.to_owned()),
        };

        self.output.record(GrepLine {
            line,
            byte_offset: context.absolute_byte_offset() as usize,
            line_number: context.line_number().expect("Have line number") as usize,
            sub_matches: vec![],
        });
        Ok(true)
    }
}

pub struct GrepNegativeMatchSink<'output> {
    output: &'output mut GrepOutput,
}

impl Sink for GrepNegativeMatchSink<'_> {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _searcher: &grep::searcher::Searcher,
        _mat: &grep::searcher::SinkMatch<'_>,
    ) -> Result<bool, Self::Error> {
        self.output.content_negated = true;
        Ok(false)
    }
}
