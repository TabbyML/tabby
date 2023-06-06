use std::collections::HashMap;

use lazy_static::lazy_static;

lazy_static! {
    static ref DEFAULT: Vec<&'static str> = vec!("\n\n");
    static ref LANGUAGES: HashMap<&'static str, Vec<&'static str>> = {
        let mut map = HashMap::new();
        map.insert("python", vec!["\n\n", "\ndef", "\n#", "\nfrom", "\nclass"]);
        map.insert(
            "javascript",
            vec!["\n\n", "\nfunction", "\n//", "\nimport", "\nclass"],
        );
        map.insert(
            "typescript",
            vec![
                "\n\n",
                "\nfunction",
                "\n//",
                "\nimport",
                "\nclass",
                "\ninterface",
                "\ntype",
            ],
        );
        map
    };
}

pub fn get_stop_words(language: &str) -> &'static Vec<&'static str> {
    LANGUAGES.get(language).unwrap_or(&DEFAULT)
}
