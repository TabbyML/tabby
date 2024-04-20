use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject, ID};
use lazy_static::lazy_static;
use strum::{EnumIter, IntoEnumIterator};

use crate::schema::Result;

#[derive(GraphQLObject, Debug, Clone)]
pub struct CompletionStats {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,

    pub language: Language,
    pub completions: i32,
    pub views: i32,
    pub selects: i32,
}

#[derive(GraphQLEnum, Clone, Debug, Eq, PartialEq, EnumIter, Hash)]
pub enum Language {
    Rust,
    Python,
    Java,
    Kotlin,
    Javascript,
    Typescript,
    Go,
    Ruby,
    CSharp,
    C,
    Cpp,
    Solidity,
    Other,
}

lazy_static! {
    static ref LANGUAGE_STRING_MAPPINGS: HashMap<Language, Vec<&'static str>> = {
        let mut map = HashMap::new();
        map.insert(Language::Rust, vec!["rust"]);
        map.insert(Language::Python, vec!["python"]);
        map.insert(Language::Java, vec!["java"]);
        map.insert(Language::Kotlin, vec!["kotlin"]);
        map.insert(Language::Javascript, vec!["javascript", "javascriptreact"]);
        map.insert(Language::Typescript, vec!["typescript", "typescriptreact"]);
        map.insert(Language::Go, vec!["go"]);
        map.insert(Language::Ruby, vec!["ruby"]);
        map.insert(Language::CSharp, vec!["csharp"]);
        map.insert(Language::C, vec!["c"]);
        map.insert(Language::Cpp, vec!["cpp", "c++"]);
        map.insert(Language::Solidity, vec!["solidity"]);
        map.insert(Language::Other, vec!["other"]);
        map
    };
    static ref STRING_LANGUAGE_MAPPINGS: HashMap<&'static str, Language> = {
        let mut map = HashMap::new();
        for (language, strings) in LANGUAGE_STRING_MAPPINGS.iter() {
            for string in strings {
                map.insert(*string, language.clone());
            }
        }
        map
    };
}

impl Language {
    pub fn all_known() -> impl Iterator<Item = Language> {
        Language::iter().filter(|l| l != &Language::Other)
    }

    pub fn to_strings(&self) -> &'static Vec<&'static str> {
        if let Some(vec) = LANGUAGE_STRING_MAPPINGS.get(self) {
            vec
        } else {
            LANGUAGE_STRING_MAPPINGS
                .get(&Language::Other)
                .expect("Language::Other should present")
        }
    }
}

impl Into<Language> for String {
    fn into(self) -> Language {
        if let Some(lang) = STRING_LANGUAGE_MAPPINGS.get(self.as_str()) {
            lang.clone()
        } else {
            Language::Other
        }
    }
}

#[async_trait]
pub trait AnalyticService: Send + Sync {
    /// Generate the report for past year, with daily granularity.
    ///
    /// `users` is a list of user IDs. If empty, the report is computed for all users.
    async fn daily_stats_in_past_year(&self, users: Vec<ID>) -> Result<Vec<CompletionStats>>;

    /// Computes the report with daily granularity.
    ///
    /// 1. [`start`, `end`) define the time range for the report.
    /// 2. `users` is a list of user IDs. If empty, the report is computed for all users.
    /// 3. `languages` is a list of programming language identifier. If empty, the report is computed for all languages.
    async fn daily_stats(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        users: Vec<ID>,
        languages: Vec<Language>,
    ) -> Result<Vec<CompletionStats>>;
}
