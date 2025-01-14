mod tokenizer;
use tantivy::{
    query::{BooleanQuery, ConstScoreQuery, Occur, Query, TermQuery},
    schema::IndexRecordOption,
    Term,
};
pub use tokenizer::tokenize_code;

use super::{corpus, IndexSchema};
use crate::api::code::CodeSearchQuery;

pub mod fields {
    // === Doc level fields ===
    /// commit ref of the file being indexed.
    pub const COMMIT: &str = "commit";

    // === Chunk level fields ===
    pub const CHUNK_GIT_URL: &str = "chunk_git_url";
    pub const CHUNK_FILEPATH: &str = "chunk_filepath";
    pub const CHUNK_LANGUAGE: &str = "chunk_language";
    pub const CHUNK_BODY: &str = "chunk_body";
    /// Optional, when None, it means this chunk contains entire content of the file.
    pub const CHUNK_START_LINE: &str = "chunk_start_line";
}

fn language_query(language: &str) -> Box<TermQuery> {
    let schema = IndexSchema::instance();
    let language = match language {
        "javascript" | "typescript" | "javascriptreact" | "typescriptreact" => {
            "javascript-typescript"
        }
        _ => language,
    };

    let mut term =
        Term::from_field_json_path(schema.field_chunk_attributes, fields::CHUNK_LANGUAGE, false);
    term.append_type_and_str(language);
    Box::new(TermQuery::new(term, IndexRecordOption::Basic))
}

pub fn body_query(tokens: &[String]) -> Box<dyn Query> {
    let schema = IndexSchema::instance();
    let subqueries: Vec<Box<dyn Query>> = tokens
        .iter()
        .map(|text| {
            let term = Term::from_field_text(schema.field_chunk_tokens, text);
            let term_query: Box<dyn Query> =
                Box::new(TermQuery::new(term, IndexRecordOption::Basic));

            term_query
        })
        .collect();

    Box::new(BooleanQuery::union(subqueries))
}

fn filepath_query(filepath: &str) -> Box<TermQuery> {
    let schema = IndexSchema::instance();
    let mut term =
        Term::from_field_json_path(schema.field_chunk_attributes, fields::CHUNK_FILEPATH, false);
    term.append_type_and_str(filepath);
    Box::new(TermQuery::new(term, IndexRecordOption::Basic))
}

pub fn code_search_query(
    query: &CodeSearchQuery,
    chunk_tokens_query: Box<dyn Query>,
) -> BooleanQuery {
    let schema = IndexSchema::instance();

    // language / git_url / filepath field shouldn't contribute to the score, mark them to 0.0.
    let mut subqueries = vec![
        (Occur::Must, chunk_tokens_query),
        (Occur::Must, schema.corpus_query(corpus::CODE)),
        (
            Occur::Must,
            Box::new(ConstScoreQuery::new(
                Box::new(schema.source_id_query(&query.source_id)),
                0.0,
            )),
        ),
    ];

    if let Some(language) = query.language.as_deref() {
        subqueries.push((
            Occur::Must,
            Box::new(ConstScoreQuery::new(language_query(language), 0.0)),
        ));
    }

    // When filepath presents, we exclude the file from the search.
    if let Some(filepath) = &query.filepath {
        subqueries.push((
            Occur::MustNot,
            Box::new(ConstScoreQuery::new(filepath_query(filepath), 0.0)),
        ))
    }

    BooleanQuery::new(subqueries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_query() {
        let lhs = language_query("javascript-typescript");
        assert_eq!(lhs.term(), language_query("javascript").term());
        assert_eq!(lhs.term(), language_query("typescript").term());
        assert_eq!(lhs.term(), language_query("typescriptreact").term());
        assert_eq!(lhs.term(), language_query("javascriptreact").term());
    }
}
