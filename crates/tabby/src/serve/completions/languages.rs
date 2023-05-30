use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

lazy_static! {
    static ref DEFAULT: Regex = Regex::new(r"(?m)\n\n").unwrap();
    static ref LANGUAGES: HashMap<&'static str, Regex> = {
        let mut map = HashMap::new();
        map.insert(
            "python",
            Regex::new(r"(?m)(\n\n|^def|^#|^from|^class)").unwrap(),
        );
        map
    };
}

pub fn remove_stop_words<'a>(language: &'a str, text: &'a str) -> &'a str {
    let re = LANGUAGES.get(language).unwrap_or(&DEFAULT);
    let position = re.find_iter(text).next();
    if let Some(m) = position {
        &text[..m.start()]
    } else {
        text
    }
}
