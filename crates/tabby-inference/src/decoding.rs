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
    pub fn create(&self, text: &str, language: &'static Language) -> StopCondition {
        StopCondition::new(self.get_re(language), text)
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
    let regex_string =
        r"(?m)\A".to_owned() + "((" + reversed_stop_words.join(")|(").as_str() + "))";
    Regex::new(&regex_string).expect("Failed to create regex")
}

pub struct StopCondition {
    stop_re: Option<Regex>,
    reversed_text: String,
}

impl StopCondition {
    pub fn new(stop_re: Option<Regex>, text: &str) -> Self {
        Self {
            stop_re,
            reversed_text: reverse(text),
        }
    }

    pub fn should_stop(&mut self, new_text: &str) -> bool {
        if !new_text.is_empty() {
            self.reversed_text = reverse(new_text) + self.reversed_text.as_str();

            if let Some(re) = &self.stop_re {
                if re.is_match(&self.reversed_text) {
                    return true;
                }
            }
        }

        false
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
