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

    /* Python */
    static ref PYTHON_STOP_WORDS: Vec<&'static str> =
        vec!["\ndef", "\n#", "\nfrom", "\nclass", "\nimport"].with_default();
    static ref PYTHON: Language = Language {
        stop_words: &PYTHON_STOP_WORDS,
        line_comment: "#",
    };

    /* Rust */
    static ref RUST_STOP_WORDS: Vec<&'static str> = vec![
        "\n//", "\nfn", "\ntrait", "\nimpl", "\nenum", "\npub", "\nextern", "\nstatic",
        "\ntrait", "\nunsafe", "\nuse"
    ]
    .with_default();
    static ref RUST: Language = Language {
        stop_words: &RUST_STOP_WORDS,
        line_comment: "//",
    };

    /* Javascript / Typescript */
    static ref JAVASCRIPT_TYPESCRIPT_STOP_WORDS: Vec<&'static str> = vec![
        "\n//",
        "\nabstract",
        "\nasync",
        "\nclass",
        "\nconst",
        "\nexport",
        "\nfunction",
        "\ninterface",
        "\nmodule",
        "\npackage",
        "\ntype",
        "\nvar",
        "\nenum",
        "\nlet",
    ]
    .with_default();
    static ref JAVASCRIPT_TYPESCRIPT: Language = Language {
        stop_words: &JAVASCRIPT_TYPESCRIPT_STOP_WORDS,
        line_comment: "//",
    };
}

pub fn get_language(language: &str) -> &'static Language {
    if language == "python" {
        &PYTHON
    } else if language == "rust" {
        &RUST
    } else if language == "javascript" || language == "typescript" {
        &JAVASCRIPT_TYPESCRIPT
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
