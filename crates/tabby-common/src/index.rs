use tantivy::{
    tokenizer::{NgramTokenizer, RegexTokenizer, RemoveLongFilter, TextAnalyzer},
    Index,
};

pub trait IndexExt {
    fn register_tokenizer(&self);
}

pub static CODE_TOKENIZER: &str = "code";
pub static IDENTIFIER_TOKENIZER: &str = "identifier";

impl IndexExt for Index {
    fn register_tokenizer(&self) {
        let code_tokenizer = TextAnalyzer::builder(RegexTokenizer::new(r"(?:\w+)").unwrap())
            .filter(RemoveLongFilter::limit(128))
            .build();

        self.tokenizers().register(CODE_TOKENIZER, code_tokenizer);

        let identifier_tokenzier =
            TextAnalyzer::builder(NgramTokenizer::prefix_only(2, 5).unwrap()).build();

        self.tokenizers()
            .register(IDENTIFIER_TOKENIZER, identifier_tokenzier);
    }
}
