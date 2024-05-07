use tantivy::{
    query::{TermQuery, TermSetQuery},
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
    /// Indexed field uniquely identifying a file in a repository, stringified SourceFileKey
    pub field_file_id: Field,
    pub field_language: Field,
    pub field_body: Field,
}

impl CodeSearchSchema {
    pub fn new() -> Self {
        let mut builder = Schema::builder();

        let code_indexing_options = TextFieldIndexing::default()
            .set_tokenizer(CODE_TOKENIZER)
            .set_index_option(tantivy::schema::IndexRecordOption::WithFreqs);
        let code_options = TextOptions::default()
            .set_indexing_options(code_indexing_options)
            .set_stored();

        let field_git_url = builder.add_text_field("git_url", STRING | STORED);
        let field_filepath = builder.add_text_field("filepath", STRING | STORED);
        let field_file_id = builder.add_text_field("file_id", STRING | STORED);
        let field_language = builder.add_text_field("language", STRING | STORED);
        let field_body = builder.add_text_field("body", code_options);
        let schema = builder.build();

        Self {
            schema,
            field_git_url,
            field_filepath,
            field_file_id,
            field_language,
            field_body,
        }
    }
}

impl Default for CodeSearchSchema {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeSearchSchema {
    pub fn language_query(&self, language: &str) -> Box<TermQuery> {
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

    pub fn body_query(&self, tokens: &[String]) -> Box<TermSetQuery> {
        Box::new(TermSetQuery::new(
            tokens
                .iter()
                .map(|x| Term::from_field_text(self.field_body, x)),
        ))
    }

    pub fn git_url_query(&self, git_url: &str) -> Box<TermQuery> {
        Box::new(TermQuery::new(
            Term::from_field_text(self.field_git_url, git_url),
            IndexRecordOption::Basic,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::CodeSearchSchema;

    #[test]
    fn test_language_query() {
        let schema = CodeSearchSchema::new();
        let lhs = schema.language_query("javascript-typescript");
        assert_eq!(lhs.term(), schema.language_query("javascript").term());
        assert_eq!(lhs.term(), schema.language_query("typescript").term());
        assert_eq!(lhs.term(), schema.language_query("typescriptreact").term());
        assert_eq!(lhs.term(), schema.language_query("javascriptreact").term());
    }
}
