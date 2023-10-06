use tantivy::{
    tokenizer::{RegexTokenizer, RemoveLongFilter, TextAnalyzer},
    Index,
};

pub trait IndexExt {
    fn register_tokenizer(&self);
}

pub static CODE_TOKENIZER: &str = "code";

impl IndexExt for Index {
    fn register_tokenizer(&self) {
        let code_tokenizer = TextAnalyzer::builder(RegexTokenizer::new(r"(?:\w+)").unwrap())
            .filter(RemoveLongFilter::limit(128))
            .build();

        self.tokenizers().register(CODE_TOKENIZER, code_tokenizer);
    }
}
