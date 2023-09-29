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

impl Default for StopWords {
    fn default() -> Self {
        Self {
            stop_regex_cache: DashMap::new(),
        }
    }
}

impl StopWords {
    pub fn create_incremental_decoding(
        &self,
        tokenizer: Arc<Tokenizer>,
        input_token_ids: &[u32],
        stop_words: &'static Vec<&'static str>,
    ) -> IncrementalDecoding {
        IncrementalDecoding::new(tokenizer, self.get_re(stop_words), input_token_ids)
    }

    fn get_re(&self, stop_words: &'static Vec<&'static str>) -> Option<Regex> {
        if stop_words.is_empty() {
            None
        } else {
            let mut re = self.stop_regex_cache.get(stop_words);
            if re.is_none() {
                self.stop_regex_cache
                    .insert(stop_words, create_stop_regex(stop_words));
                re = self.stop_regex_cache.get(stop_words);
            }
            re.map(|x| x.value().clone())
        }
    }
}

fn create_stop_regex(stop_words: &[&str]) -> Regex {
    let tokens: Vec<String> = stop_words.iter().map(reverse).collect();

    // (?m) enables multi-line matching mode.
    // \A means absolute begins of string.
    let regex_string = r"(?m)\A".to_owned() + &tokens.join("|");
    Regex::new(&regex_string).unwrap()
}

pub struct IncrementalDecoding {
    tokenizer: Arc<Tokenizer>,
    stop_re: Option<Regex>,

    token_ids: Vec<u32>,
    text: String,
}

impl IncrementalDecoding {
    pub fn new(tokenizer: Arc<Tokenizer>, stop_re: Option<Regex>, input_token_ids: &[u32]) -> Self {
        let text = tokenizer
            .decode(input_token_ids, /* skip_special_token = */ true)
            .expect("Cannot decode token from tokenizer.");
        Self {
            tokenizer,
            stop_re,
            token_ids: input_token_ids.to_owned(),
            text,
        }
    }

    pub fn next_token(&mut self, token_id: u32) -> Option<String> {
        self.token_ids.push(token_id);
        let text = self
            .tokenizer
            .decode(&self.token_ids, /* skip_special_token = */ true)
            .expect("Cannot decode token from tokenizer.")
            .as_bytes()
            .to_vec();
        let text: String = unsafe { String::from_utf8_unchecked(text) };
        let reversed_text = reverse(&text.as_str());

        if let Some(re) = &self.stop_re {
            if re.find(&reversed_text).is_some() {
                return None;
            }
        }

        let new_text = if text.ends_with('�') {
            "".to_owned()
        } else {
            text[self.text.len()..].to_owned()
        };

        self.text = text;
        println!("text {}", new_text);
        Some(new_text)
    }
}
