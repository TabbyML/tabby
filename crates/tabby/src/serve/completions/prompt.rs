use std::collections::HashMap;

use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use strfmt::strfmt;
use tabby_common::path::index_dir;
use tantivy::{
    collector::TopDocs, query::QueryParser, schema::Field, Index, ReloadPolicy, Searcher,
};
use tracing::warn;

use super::Segments;

pub struct PromptBuilder {
    prompt_template: Option<String>,
    index: Option<IndexState>,
}

impl PromptBuilder {
    pub fn new(prompt_template: Option<String>) -> Self {
        let index = IndexState::new();
        if let Err(err) = &index {
            warn!("Failed to open index in {:?}: {:?}", index_dir(), err);
        }

        PromptBuilder {
            prompt_template,
            index: index.ok(),
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
        println!("segments\n{}", segments.prefix);
        if let Some(suffix) = segments.suffix {
            self.build_prompt(segments.prefix, suffix)
        } else {
            self.build_prompt(segments.prefix, "".to_owned())
        }
    }

    fn rewrite(&self, language: &str, segments: Segments) -> Segments {
        if let Some(index) = &self.index {
            rewrite_with_index(index, language, segments)
        } else {
            segments
        }
    }
}

fn rewrite_with_index(index: &IndexState, language: &str, segments: Segments) -> Segments {
    let snippets = collect_snippets(index, language, &segments.prefix);
    if snippets.is_empty() {
        segments
    } else {
        let prefix = build_prefix(language, &segments.prefix, snippets);
        Segments { prefix, ..segments }
    }
}

fn build_prefix(language: &str, prefix: &str, snippets: Vec<String>) -> String {
    let comment_char = LANGUAGE_LINE_COMMENT_CHAR.get(language).unwrap();
    let mut lines: Vec<String> =
        vec!["Below are some relevant code snippets found in the repository:".to_owned()];

    for (i, snippet) in snippets.iter().enumerate() {
        lines.push(format!("== Snippet {} ==", i + 1));
        for line in snippet.lines() {
            lines.push(line.to_owned());
        }
    }

    let commented_lines: Vec<String> = lines
        .iter()
        .map(|x| format!("{} {}", comment_char, x))
        .collect();
    let comments = commented_lines.join("\n");
    format!("{}\n{}", comments, prefix)
}

fn collect_snippets(index: &IndexState, language: &str, text: &str) -> Vec<String> {
    let mut ret = Vec::new();
    let sanitized_text = sanitize_text(text);
    if sanitized_text.is_empty() {
        return ret;
    }

    let query_text = format!(
        "language:{} AND kind:call AND ({})",
        language, sanitized_text
    );
    let query = match index.query_parser.parse_query(&query_text) {
        Ok(query) => query,
        Err(err) => {
            warn!("Failed to parse query: {}", err);
            return ret;
        }
    };

    let top_docs = index
        .searcher
        .search(&query, &TopDocs::with_limit(5))
        .unwrap();
    for (_score, doc_address) in top_docs {
        let doc = index.searcher.doc(doc_address).unwrap();
        if let Some(body) = doc.get_first(index.field_body) {
            if let Some(snippet) = body.as_text() {
                ret.push(snippet.to_owned());
            }
        }
    }

    ret
}

fn sanitize_text(text: &str) -> String {
    let x = text.replace(|c: char| !c.is_ascii_digit() && !c.is_alphabetic(), " ");
    let tokens: Vec<&str> = x.split(' ').collect();
    tokens.join(" ")
}

struct IndexState {
    searcher: Searcher,
    query_parser: QueryParser,
    field_body: Field,
}

impl IndexState {
    fn new() -> Result<IndexState> {
        let index = Index::open_in_dir(index_dir())?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommit)
            .try_into()?;
        let field_name = index
            .schema()
            .get_field("name")
            .ok_or(anyhow!("Index doesn't have required field"))?;
        let field_body = index
            .schema()
            .get_field("body")
            .ok_or(anyhow!("Index doesn't have required field"))?;
        let query_parser = QueryParser::for_index(&index, vec![field_name]);
        Ok(Self {
            searcher: reader.searcher(),
            query_parser,
            field_body,
        })
    }
}

lazy_static! {
    static ref LANGUAGE_LINE_COMMENT_CHAR: HashMap<&'static str, &'static str> =
        HashMap::from([("python", "#")]);
}
