mod tokenizer;
use tantivy::{
    query::{BooleanQuery, BoostQuery, ConstScoreQuery, Occur, Query, TermQuery},
    schema::IndexRecordOption,
    Term,
};
pub use tokenizer::tokenize_code;

use super::IndexSchema;
use crate::api::code::CodeSearchQuery;

pub mod fields {
    pub const CHUNK_GIT_URL: &str = "chunk_git_url";
    pub const CHUNK_FILEPATH: &str = "chunk_filepath";
    pub const CHUNK_LANGUAGE: &str = "chunk_language";
    pub const CHUNK_BODY: &str = "chunk_body";
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

fn body_query(tokens: &[String]) -> Box<dyn Query> {
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

fn git_url_query(git_url: &str) -> Box<TermQuery> {
    let schema = IndexSchema::instance();
    let mut term =
        Term::from_field_json_path(schema.field_chunk_attributes, fields::CHUNK_GIT_URL, false);
    term.append_type_and_str(git_url);
    Box::new(TermQuery::new(term, IndexRecordOption::Basic))
}

fn filepath_query(filepath: &str) -> Box<TermQuery> {
    let schema = IndexSchema::instance();
    let mut term =
        Term::from_field_json_path(schema.field_chunk_attributes, fields::CHUNK_FILEPATH, false);
    term.append_type_and_str(filepath);
    Box::new(TermQuery::new(term, IndexRecordOption::Basic))
}

pub fn code_search_query(query: &CodeSearchQuery) -> BooleanQuery {
    let language_query = language_query(&query.language);
    let git_url_query = git_url_query(&query.git_url);

    // Create body query with a scoring normalized by the number of tokens.
    let body_tokens = tokenize_code(&query.content);
    let body_query = body_query(&body_tokens);
    let normalized_score_body_query = BoostQuery::new(body_query, 1.0 / body_tokens.len() as f32);

    // language / git_url / filepath field shouldn't contribute to the score, mark them to 0.0.
    let mut subqueries: Vec<(Occur, Box<dyn Query>)> = vec![
        (
            Occur::Must,
            Box::new(ConstScoreQuery::new(language_query, 0.0)),
        ),
        (Occur::Must, Box::new(normalized_score_body_query)),
        (
            Occur::Must,
            Box::new(ConstScoreQuery::new(git_url_query, 0.0)),
        ),
    ];

    // When filepath presents, we exlucde the file from the search.
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
