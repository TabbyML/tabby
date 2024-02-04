use dashmap::DashMap;
use tabby_common::languages::Language;
use trie_rs::{Trie, TrieBuilder};

pub struct StopConditionFactory {
    stop_trie_cache: DashMap<String, Trie<u8>>,
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
            stop_trie_cache: DashMap::new(),
        }
    }
}

type CachedTrie<'a> = dashmap::mapref::one::Ref<'a, String, Trie<u8>>;

impl StopConditionFactory {
    pub fn create(
        &self,
        text: &str,
        max_decoding_length: usize,
        language: Option<&'static Language>,
    ) -> StopCondition {
        if let Some(language) = language {
            StopCondition::new(self.get_trie(language), max_decoding_length, text)
        } else {
            StopCondition::new(None, max_decoding_length, text)
        }
    }

    fn get_trie<'a>(&'a self, language: &'static Language) -> Option<CachedTrie<'a>> {
        let stop_words = language.get_stop_words();
        if stop_words.is_empty() {
            None
        } else {
            let hashkey = language.get_hashkey();
            let mut trie = self.stop_trie_cache.get(&hashkey);
            if trie.is_none() {
                self.stop_trie_cache
                    .insert(hashkey.clone(), create_stop_trie(stop_words));
                trie = self.stop_trie_cache.get(&hashkey);
            }

            trie
        }
    }
}

fn create_stop_trie(stop_words: Vec<String>) -> Trie<u8> {
    let mut builder = TrieBuilder::new();
    for word in stop_words {
        builder.push(reverse(word))
    }
    builder.build()
}

pub struct StopCondition<'a> {
    stop_trie: Option<CachedTrie<'a>>,
    max_decoding_length: usize,
    reversed_text: String,
    num_decoded: usize,
}

impl<'a> StopCondition<'a> {
    pub fn new(stop_trie: Option<CachedTrie<'a>>, max_decoding_length: usize, text: &str) -> Self {
        Self {
            stop_trie,
            max_decoding_length,
            reversed_text: reverse(text),
            num_decoded: 0,
        }
    }

    pub fn should_stop(&mut self, new_text: &str) -> (bool, usize) {
        if !new_text.is_empty() {
            self.reversed_text = reverse(new_text) + &self.reversed_text;

            if let Some(re) = &self.stop_trie {
                let matches = re.common_prefix_search(&self.reversed_text);
                let matched_length = matches.into_iter().map(|x| x.len()).max();
                if let Some(matched_length) = matched_length {
                    return (true, matched_length);
                } else {
                    return (false, 0);
                };
            }
        }

        self.num_decoded += 1;
        (self.num_decoded >= self.max_decoding_length, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_works() {
        let text = reverse("void write_u32(std::uint32_t val) const {\n        write_raw(&val, sizeof(val));\n    }\n\n    ~llama_file() {\n        if (fp) {\n            std::fclose(fp);\n        }\n    }\n};\n\nvoid");

        let trie = create_stop_trie(vec!["\n\n".to_owned(), "\n\n  ".to_owned()]);
        assert!(trie.common_prefix_search(&text).is_empty());

        let trie = create_stop_trie(vec![
            "\n\n".to_owned(),
            "\n\n  ".to_owned(),
            "\nvoid".to_owned(),
        ]);
        assert!(!trie.common_prefix_search(&text).is_empty());
    }
}
