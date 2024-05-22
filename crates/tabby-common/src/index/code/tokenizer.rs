use tantivy::{
    tokenizer::{
        RegexTokenizer, RemoveLongFilter, TextAnalyzer, TokenStream, Tokenizer,
    },
    Index,
};

pub static CODE_TOKENIZER: &str = "code";

pub fn register_tokenizers(index: &Index) {
    index
        .tokenizers()
        .register(CODE_TOKENIZER, make_code_tokenizer());
}

pub fn tokenize_code(text: &str) -> Vec<String> {
    let mut code_tokenizer = make_code_tokenizer();
    let mut tokens = vec![];

    let mut token_stream = code_tokenizer.token_stream(text);
    while let Some(token) = token_stream.next() {
        tokens.push(token.text.to_owned());
    }

    tokens
}

fn make_code_tokenizer() -> TextAnalyzer {
    TextAnalyzer::builder(RegexTokenizer::new(r"(?:\w+)").unwrap())
        .filter(RemoveLongFilter::limit(64))
        .build()
}
