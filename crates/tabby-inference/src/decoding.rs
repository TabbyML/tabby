use std::sync::Arc;

use dashmap::DashMap;
use regex::Regex;
use tabby_common::languages::Language;
use tokenizers::tokenizer::Tokenizer;

pub struct DecodingFactory {
    stop_regex_cache: DashMap<String, Regex>,
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
        language: &'static Language,
    ) -> IncrementalDecoding {
        IncrementalDecoding::new(tokenizer, self.get_re(language), input_token_ids)
    }

    fn get_re(&self, language: &'static Language) -> Option<Regex> {
        let stop_words = language.get_stop_words();
        if stop_words.is_empty() {
            None
        } else {
            let hashkey = language.get_hashkey();
            let mut re = self.stop_regex_cache.get(&hashkey);
            if re.is_none() {
                self.stop_regex_cache
                    .insert(hashkey.clone(), create_stop_regex(stop_words));
                re = self.stop_regex_cache.get(&hashkey);
            }
            re.map(|x| x.value().clone())
        }
    }
}

fn create_stop_regex(stop_words: Vec<String>) -> Regex {
    // (?m) enables multi-line matching mode.
    // \A means absolute begins of string.
    let reversed_stop_words: Vec<_> = stop_words
        .iter()
        .map(|x| regex::escape(&reverse(x)))
        .collect();
    let regex_string = r"(?m)\A".to_owned() + "((" + &reversed_stop_words.join(")|(") + "))";
    Regex::new(&regex_string).expect("Failed to create regex")
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
                if re.is_match(&self.reversed_text) {
                    return None;
                }
            }
        }

        Some(new_text.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let text = reverse("void write_u32(std::uint32_t val) const {\n        write_raw(&val, sizeof(val));\n    }\n\n    ~llama_file() {\n        if (fp) {\n            std::fclose(fp);\n        }\n    }\n};\n\nvoid");
        assert!(!create_stop_regex(vec!["\n\n".to_owned(), "\n\n  ".to_owned()]).is_match(&text));
        assert!(create_stop_regex(vec![
            "\n\n".to_owned(),
            "\n\n  ".to_owned(),
            "\nvoid".to_owned()
        ])
        .is_match(&text));
    }
}
