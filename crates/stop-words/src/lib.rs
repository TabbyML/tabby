use dashmap::DashMap;
use regex::Regex;
use tokenizers::tokenizer::Tokenizer;

pub struct StopWords {
    stop_regex_cache: DashMap<&'static Vec<&'static str>, Regex>,
}

fn reverse(s: &String) -> String {
    // Special treatment for byte fallback token.
    // https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/decoders/byte_fallback.rs
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        // Keep byte fallback tokens like <0x0A> as is, do not reverse it.
        // This won't really affect stop words regex logic, but brings more readability when
        // debugging decoding steps.
        s.to_owned()
    } else {
        s.chars().rev().collect()
    }
}

impl StopWords {
    pub fn create_condition(
        &mut self,
        tokenizer: &Tokenizer,
        stop_words: &'static Vec<&'static str>,
        stop_words_encoding_offset: Option<usize>,
    ) -> StopWordsCondition {
        let mut re = self.stop_regex_cache.get(stop_words);
        if re.is_none() {
            self.stop_regex_cache.insert(
                stop_words,
                create_stop_regex(tokenizer, stop_words, stop_words_encoding_offset),
            );
            re = self.stop_regex_cache.get(stop_words);
        }

        StopWordsCondition::new(re.map(|x| x.value().clone()))
    }
}

fn create_stop_regex(
    tokenizer: &Tokenizer,
    stop_words: &[&str],
    stop_words_encoding_offset: Option<usize>,
) -> Regex {
    let encodings = tokenizer
        .encode_batch(stop_words.to_owned(), false)
        .unwrap();
    let stop_tokens: Vec<String> = encodings
        .iter()
        .map(|x| {
            x.get_tokens()[stop_words_encoding_offset.unwrap_or(0)..]
                .iter()
                .rev()
                .map(reverse)
                .collect::<Vec<String>>()
                .join("")
        })
        .collect();

    // (?m) enables multi-line matching mode.
    // \A means absolute begins of string.
    let regex_string = r"(?m)\A".to_owned() + &stop_tokens.join("|");
    Regex::new(&regex_string).unwrap()
}

pub struct StopWordsCondition {
    stop_re: Option<Regex>,
    reversed_output_text: String,
}

impl StopWordsCondition {
    pub fn new(stop_re: Option<Regex>) -> Self {
        Self {
            stop_re,
            reversed_output_text: String::new(),
        }
    }

    pub fn next_token(&mut self, token: String) -> bool {
        if let Some(re) = &self.stop_re {
            let mut new_token = reverse(&token);
            new_token.push_str(&self.reversed_output_text);
            self.reversed_output_text = new_token;
            re.find(&self.reversed_output_text).is_some()
        } else {
            false
        }
    }
}
