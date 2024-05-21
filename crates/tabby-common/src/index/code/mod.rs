mod document;

use lazy_static::lazy_static;
use regex::Regex;
use tantivy::{
    query::{BooleanQuery, BoostQuery, ConstScoreQuery, Occur, Query, TermQuery},
    schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, STORED, STRING},
    tokenizer::{RegexTokenizer, RemoveLongFilter, TextAnalyzer},
    Index, Term,
};

use crate::api::code::CodeSearchQuery;

static CODE_TOKENIZER: &str = "code";

pub fn register_tokenizers(index: &Index) {
    let code_tokenizer = TextAnalyzer::builder(RegexTokenizer::new(r"(?:\w+)").unwrap())
        .filter(RemoveLongFilter::limit(64))
        .build();

    index.tokenizers().register(CODE_TOKENIZER, code_tokenizer);
}

pub struct CodeSearchSchema {
    pub schema: Schema,
    pub field_git_url: Field,
    pub field_filepath: Field,
    pub field_file_id: Field,
    pub field_language: Field,
    pub field_body: Field,
    pub field_start_line: Field,
}

impl CodeSearchSchema {
    fn new() -> Self {
        let mut builder = Schema::builder();

        let code_indexing_options = TextFieldIndexing::default()
            .set_tokenizer(CODE_TOKENIZER)
            .set_index_option(tantivy::schema::IndexRecordOption::WithFreqs);
        let code_options = TextOptions::default()
            .set_indexing_options(code_indexing_options)
            .set_stored();

        let field_git_url = builder.add_text_field("git_url", STRING | STORED);
        let field_filepath = builder.add_text_field("filepath", STRING | STORED);
        let field_source_file_key = builder.add_text_field("file_id", STRING | STORED);
        let field_language = builder.add_text_field("language", STRING | STORED);
        let field_body = builder.add_text_field("body", code_options);
        let field_start_line = builder.add_u64_field("start_line", STORED);
        let schema = builder.build();

        Self {
            schema,
            field_git_url,
            field_filepath,
            field_file_id: field_source_file_key,
            field_language,
            field_body,
            field_start_line,
        }
    }

    pub fn instance() -> &'static Self {
        &CODE_SEARCH_SCHEMA
    }
}

lazy_static! {
    static ref CODE_SEARCH_SCHEMA: CodeSearchSchema = CodeSearchSchema::new();
    static ref BODY_TOKENIZER: Regex = Regex::new(r"[^\w]").unwrap();
}

impl CodeSearchSchema {
    fn language_query(&self, language: &str) -> Box<TermQuery> {
        let language = match language {
            "javascript" | "typescript" | "javascriptreact" | "typescriptreact" => {
                "javascript-typescript"
            }
            _ => language,
        };

        Box::new(TermQuery::new(
            Term::from_field_text(self.field_language, language),
            IndexRecordOption::Basic,
        ))
    }

    fn body_query(&self, tokens: &[String]) -> Box<dyn Query> {
        Box::new(BooleanQuery::new_multiterms_query(
            tokens
                .iter()
                .map(|x| Term::from_field_text(self.field_body, x))
                .collect(),
        ))
    }

    fn git_url_query(&self, git_url: &str) -> Box<TermQuery> {
        Box::new(TermQuery::new(
            Term::from_field_text(self.field_git_url, git_url),
            IndexRecordOption::Basic,
        ))
    }

    fn filepath_query(&self, filepath: &str) -> Box<TermQuery> {
        Box::new(TermQuery::new(
            Term::from_field_text(self.field_filepath, filepath),
            IndexRecordOption::Basic,
        ))
    }

    pub fn tokenize_body(text: &str) -> Vec<String> {
        BODY_TOKENIZER
            .split(text)
            .filter(|x| !x.is_empty())
            .map(|x| x.to_owned())
            .collect()
    }

    pub fn code_search_query(&self, query: &CodeSearchQuery) -> BooleanQuery {
        let language_query = self.language_query(&query.language);
        let git_url_query = self.git_url_query(&query.git_url);

        let body_tokens = CodeSearchSchema::tokenize_body(&query.content);
        let body_query = self.body_query(&body_tokens);
        let normalized_score_body_query =
            BoostQuery::new(body_query, 1.0 / body_tokens.len() as f32);

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
                Box::new(ConstScoreQuery::new(self.filepath_query(filepath), 0.0)),
            ))
        }

        BooleanQuery::new(subqueries)
    }
}

#[cfg(test)]
mod tests {
    use super::CodeSearchSchema;

    #[test]
    fn test_language_query() {
        let schema = CodeSearchSchema::instance();
        let lhs = schema.language_query("javascript-typescript");
        assert_eq!(lhs.term(), schema.language_query("javascript").term());
        assert_eq!(lhs.term(), schema.language_query("typescript").term());
        assert_eq!(lhs.term(), schema.language_query("typescriptreact").term());
        assert_eq!(lhs.term(), schema.language_query("javascriptreact").term());
    }

    /// Empty strings tokens are not participating rag search and therefore could be removed.
    #[test]
    fn test_tokenized_text_filter() {
        let prefix = r#"public static String getFileExtension(String fullName) {
        String fileName = (new File(fullName)).getName();
        int dotIndex = fileName.lastIndexOf('.');
         }"#;

        // with filter
        assert_eq!(
            CodeSearchSchema::tokenize_body(prefix),
            [
                "public",
                "static",
                "String",
                "getFileExtension",
                "String",
                "fullName",
                "String",
                "fileName",
                "new",
                "File",
                "fullName",
                "getName",
                "int",
                "dotIndex",
                "fileName",
                "lastIndexOf",
            ]
        );
    }
}
