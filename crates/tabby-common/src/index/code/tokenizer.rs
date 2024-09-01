use lazy_static::lazy_static;
use tantivy::tokenizer::{RegexTokenizer, RemoveLongFilter, TextAnalyzer, TokenStream};

pub fn tokenize_code(text: &str) -> Vec<String> {
    let mut code_tokenizer = make_code_tokenizer();
    let mut tokens = vec![];

    let mut token_stream = code_tokenizer.token_stream(text);
    while let Some(token) = token_stream.next() {
        tokens.push(token.text.to_owned());
    }

    tokens
}

lazy_static! {
    static ref CODE_TOKENIZER: TextAnalyzer = {
        TextAnalyzer::builder(RegexTokenizer::new(r"(?:\w+)").unwrap())
            .filter(RemoveLongFilter::limit(64))
            .build()
    };
}

fn make_code_tokenizer() -> TextAnalyzer {
    CODE_TOKENIZER.clone()
}

#[cfg(test)]
mod tests {
    use crate::index::code::tokenizer::tokenize_code;

    /// Empty strings tokens are not participating rag search and therefore could be removed.
    #[test]
    fn test_tokenize_code() {
        let prefix = r#"public static String getFileExtension(String this_is_an_underscore_name) {
        String fileName = (new File(this_is_an_underscore_name)).getName();
        int dotIndex = fileName.lastIndexOf('.');
         }"#;

        // with filter
        assert_eq!(
            tokenize_code(prefix),
            [
                "public",
                "static",
                "String",
                "getFileExtension",
                "String",
                "this_is_an_underscore_name",
                "String",
                "fileName",
                "new",
                "File",
                "this_is_an_underscore_name",
                "getName",
                "int",
                "dotIndex",
                "fileName",
                "lastIndexOf",
            ]
        );
    }
}
