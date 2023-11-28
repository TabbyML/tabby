use dashmap::DashMap;
use regex::Regex;
use tabby_common::languages::Language;

pub struct StopConditionFactory {
    stop_regex_cache: DashMap<String, Regex>,
}

fn reverse<T>(s: T) -> String
where
    T: Into<String>,
{
    s.into().chars().rev().collect()
}

impl Default for StopConditionFactory {
    fn default() -> Self {
        Self {
            stop_regex_cache: DashMap::new(),
        }
    }
}

impl StopConditionFactory {
    pub fn create(
        &self,
        text: &str,
        max_decoding_length: usize,
        language: Option<&'static Language>,
    ) -> StopCondition {
        if let Some(language) = language {
            StopCondition::new(self.get_re(language), max_decoding_length, text)
        } else {
            StopCondition::new(None, max_decoding_length, text)
        }
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

    pub fn trim_stop_words(&self, language: &'static Language, text: &str) -> Option<String> {
        let Some(re) = self.get_re(language) else {
            return None;
        };

        let text = reverse(text);

        let text = if let Some(m) = re.find_at(&text, 0) {
            &text[m.end()..]
        } else {
            &text
        };

        Some(reverse(text))
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

pub struct StopCondition {
    stop_re: Option<Regex>,
    max_decoding_length: usize,
    reversed_text: String,
    num_decoded: usize,
}

impl StopCondition {
    pub fn new(stop_re: Option<Regex>, max_decoding_length: usize, text: &str) -> Self {
        Self {
            stop_re,
            max_decoding_length,
            reversed_text: reverse(text),
            num_decoded: 0,
        }
    }

    pub fn should_stop(&mut self, new_text: &str) -> bool {
        if !new_text.is_empty() {
            self.reversed_text = reverse(new_text) + &self.reversed_text;

            if let Some(re) = &self.stop_re {
                if re.is_match(&self.reversed_text) {
                    return true;
                }
            }
        }

        self.num_decoded += 1;
        self.num_decoded >= self.max_decoding_length
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
