use lazy_static::lazy_static;
use tantivy::{
    query::{BooleanQuery, ConstScoreQuery, Query, TermQuery},
    schema::{Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, STORED, STRING},
    tokenizer::{RegexTokenizer, RemoveLongFilter, TextAnalyzer},
    Index, Term,
};

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

    pub fn code_search_query(
        &self,
        git_url: &str,
        language: &str,
        tokens: &[String],
    ) -> BooleanQuery {
        let language_query = self.language_query(language);
        let body_query = self.body_query(tokens);
        let git_url_query = self.git_url_query(git_url);

        // language / git_url field shouldn't contribute to the score, mark them to 0.0.
        BooleanQuery::new(vec![
            (
                tantivy::query::Occur::Must,
                Box::new(ConstScoreQuery::new(language_query, 0.0)),
            ),
            (tantivy::query::Occur::Must, body_query),
            (
                tantivy::query::Occur::Must,
                Box::new(ConstScoreQuery::new(git_url_query, 0.0)),
            ),
        ])
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
}
