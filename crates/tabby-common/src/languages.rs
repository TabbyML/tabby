use lazy_static::lazy_static;
use serde::Deserialize;

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
}

#[derive(Deserialize)]
struct ConfigList {
    config: Vec<Language>,
}

#[derive(Deserialize, Debug)]
pub struct Language {
    languages: Vec<String>,
    top_level_keywords: Vec<String>,

    pub line_comment: String,
}

impl Language {
    pub fn get_stop_words(&self) -> Vec<String> {
        let mut out = vec![];
        out.push(format!("\n{}", self.line_comment));
        for word in &self.top_level_keywords {
            out.push(format!("\n{}", word));
        }

        for x in DEFAULT.iter() {
            out.push((*x).to_owned());
        }

        out
    }

    pub fn get_hashkey(&self) -> String {
        self.languages[0].clone()
    }
}

lazy_static! {
    static ref CONFIG: ConfigList =
        serdeconv::from_toml_str(include_str!("../assets/languages.toml")).unwrap();
    pub static ref UNKNOWN_LANGUAGE: Language = Language {
        languages: vec!["unknown".to_owned()],
        line_comment: "".to_owned(),
        top_level_keywords: vec![],
    };
}

pub fn get_language(language: &str) -> &'static Language {
    CONFIG
        .config
        .iter()
        .find(|c| c.languages.iter().any(|x| x == language))
        .unwrap_or(&UNKNOWN_LANGUAGE)
}
