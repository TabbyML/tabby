use std::sync::Arc;

use dashmap::DashMap;
use regex::Regex;
use tokenizers::tokenizer::Tokenizer;

pub struct DecodingFactory {
    stop_regex_cache: DashMap<&'static Vec<&'static str>, Regex>,
}

fn reverse<T>(s: T) -> String
where
    T: Into<String>,
{
    s.into().chars().rev().collect()
}

impl Default for DecodingFactory {
    fn default() -> Self {
        Self {
            stop_regex_cache: DashMap::new(),
        }
    }
}

impl DecodingFactory {
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
    let tokens: Vec<String> = stop_words.iter().map(|x| reverse(*x)).collect();

    // (?m) enables multi-line matching mode.
    // \A means absolute begins of string.
    let regex_string = r"(?m)\A".to_owned() + &tokens.join("|");
    Regex::new(&regex_string).unwrap()
}

pub struct IncrementalDecoding {
    tokenizer: Arc<Tokenizer>,
    stop_re: Option<Regex>,

    token_ids: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,

    reversed_text: String,
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
            prefix_offset: 0,
            read_offset: input_token_ids.len(),
            reversed_text: reverse(text),
        }
    }

    pub fn next_token(&mut self, token_id: u32) -> Option<String> {
        let skip_special_token = true;
        self.token_ids.push(token_id);

        let prefix_text = self
            .tokenizer
            .decode(
                &self.token_ids[self.prefix_offset..self.read_offset],
                skip_special_token,
            )
            .expect("Cannot decode token from tokenizer.");

        let new_text = self
            .tokenizer
            .decode(&self.token_ids[self.prefix_offset..], skip_special_token)
            .expect("Cannot decode token from tokenizer.");

        let new_text = if new_text.len() > prefix_text.len() && !new_text.ends_with('�') {
            self.prefix_offset = self.read_offset;
            self.read_offset = self.token_ids.len();
            &new_text[prefix_text.len()..]
        } else {
            ""
        };

        if !new_text.is_empty() {
            self.reversed_text = reverse(new_text) + &self.reversed_text;

            if let Some(re) = &self.stop_re {
                if re.find(&self.reversed_text).is_some() {
                    return None;
                }
            }
        }

        Some(new_text.to_owned())
    }
}
