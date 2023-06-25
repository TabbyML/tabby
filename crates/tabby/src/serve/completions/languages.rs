use std::collections::HashMap;

use lazy_static::lazy_static;

lazy_static! {
    static ref DEFAULT: Vec<&'static str> = vec![
        "\n\n",
        "\n\n  ",
        "\n\n    ",
        "\n\n      ",
        "\n\n        ",
        "\n\n          ",
        "\n\n            ",
        "\n\n              ",
        "\n\n",
        "\n\n\t",
        "\n\n\t\t",
        "\n\n\t\t\t",
        "\n\n\t\t\t\t",
        "\n\n\t\t\t\t\t",
        "\n\n\t\t\t\t\t\t",
        "\n\n\t\t\t\t\t\t\t",
    ];
    static ref LANGUAGES: HashMap<&'static str, Vec<&'static str>> = {
        let mut map = HashMap::new();
        map.insert(
            "python",
            vec!["\ndef", "\n#", "\nfrom", "\nclass"].with_default(),
        );
        map.insert(
            "javascript",
            vec!["\nfunction", "\n//", "\nimport", "\nclass"],
        );
        map.insert(
            "typescript",
            vec![
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

trait WithDefault {
    fn with_default(self) -> Self;
}

impl WithDefault for Vec<&'static str> {
    fn with_default(mut self) -> Self {
        let mut x = DEFAULT.clone();
        self.append(&mut x);
        self
    }
}

pub fn get_stop_words(language: &str) -> &'static Vec<&'static str> {
    LANGUAGES.get(language).unwrap_or(&DEFAULT)
}
