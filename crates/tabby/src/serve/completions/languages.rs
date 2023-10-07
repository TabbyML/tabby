use lazy_static::lazy_static;

pub struct Language {
    pub stop_words: &'static [&'static str],
    pub line_comment: &'static str,
}

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
        "\n\n\t",
        "\n\n\t\t",
        "\n\n\t\t\t",
        "\n\n\t\t\t\t",
        "\n\n\t\t\t\t\t",
        "\n\n\t\t\t\t\t\t",
        "\n\n\t\t\t\t\t\t\t",
    ];
    static ref UNKONWN: Language = Language {
        stop_words: &DEFAULT,
        line_comment: "#"
    };
    static ref PYTHON_STOP_WORDS: Vec<&'static str> =
        vec!["\ndef", "\n#", "\nfrom", "\nclass"].with_default();
    static ref PYTHON: Language = Language {
        stop_words: &PYTHON_STOP_WORDS,
        line_comment: "#",
    };
    static ref RUST_STOP_WORDS: Vec<&'static str> =
        vec!["\ndef", "\n#", "\nfrom", "\nclass"].with_default();
    static ref RUST: Language = Language {
        stop_words: &RUST_STOP_WORDS,
        line_comment: "//",
    };
    static ref JAVASCRIPT_STOP_WORDS: Vec<&'static str> =
        vec!["\ndef", "\n#", "\nfrom", "\nclass"].with_default();
    static ref JAVASCRIPT: Language = Language {
        stop_words: &JAVASCRIPT_STOP_WORDS,
        line_comment: "",
    };
    static ref TYPESCRIPT_STOP_WORDS: Vec<&'static str> =
        vec!["\ndef", "\n#", "\nfrom", "\nclass"].with_default();
    static ref TYPESCRIPT: Language = Language {
        stop_words: &TYPESCRIPT_STOP_WORDS,
        line_comment: "",
    };
}

pub fn get_language(language: &str) -> &'static Language {
    if language == "python" {
        &PYTHON
    } else if language == "rust" {
        &RUST
    } else if language == "javascript" {
        &JAVASCRIPT
    } else if language == "typescript" {
        &TYPESCRIPT
    } else {
        &UNKONWN
    }
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
