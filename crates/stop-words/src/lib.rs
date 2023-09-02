use std::sync::Arc;

use dashmap::DashMap;
use regex::Regex;
use tokenizers::tokenizer::Tokenizer;

pub struct StopWords {
    stop_regex_cache: DashMap<&'static Vec<&'static str>, Regex>,
}

fn reverse(s: &&str) -> String {
    s.chars().rev().collect()
}

impl StopWords {
    pub fn new() -> Self {
        Self {
            stop_regex_cache: DashMap::new(),
        }
    }

    pub fn create_condition(
        &self,
        tokenizer: Arc<Tokenizer>,
        stop_words: &'static Vec<&'static str>,
        stop_words_encoding_offset: Option<usize>,
    ) -> StopWordsCondition {
        let re = if stop_words.is_empty() {
            None
        } else {
            let mut re = self.stop_regex_cache.get(stop_words);
            if re.is_none() {
                self.stop_regex_cache.insert(
                    stop_words,
                    create_stop_regex(stop_words),
                );
                re = self.stop_regex_cache.get(stop_words);
            }
            re.map(|x| x.value().clone())
        };

        StopWordsCondition::new(tokenizer, re)
    }
}

fn create_stop_regex(stop_words: &[&str]) -> Regex {
    let tokens: Vec<String> = stop_words.iter().map(reverse).collect();

    // (?m) enables multi-line matching mode.
    // \A means absolute begins of string.
    let regex_string = r"(?m)\A".to_owned() + &tokens.join("|");
    Regex::new(&regex_string).unwrap()
}

pub struct StopWordsCondition {
    tokenizer: Arc<Tokenizer>,
    stop_re: Option<Regex>,
    reversed_output_text: String,
}

impl StopWordsCondition {
    pub fn new(tokenizer: Arc<Tokenizer>, stop_re: Option<Regex>) -> Self {
        Self {
            tokenizer,
            stop_re,
            reversed_output_text: String::new(),
        }
    }

    pub fn next_token(&mut self, token_id: u32) -> bool {
        if let Some(re) = &self.stop_re {
            let token = self.tokenizer.decode(vec![token_id], false).unwrap();
            let mut new_token = reverse(&token.as_str());
            new_token.push_str(&self.reversed_output_text);
            self.reversed_output_text = new_token;
            re.find(&self.reversed_output_text).is_some()
        } else {
            false
        }
    }
}
