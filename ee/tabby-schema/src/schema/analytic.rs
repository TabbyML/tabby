use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject, ID};
use lazy_static::lazy_static;
use strum::{EnumIter, IntoEnumIterator};

use crate::schema::Result;

#[derive(GraphQLObject)]
pub struct DiskUsageStats {
    pub events: DiskUsage,
    pub indexed_repositories: DiskUsage,
    pub database: DiskUsage,
    pub models: DiskUsage,
}

#[derive(GraphQLObject)]
pub struct DiskUsage {
    pub filepath: Vec<String>,

    /// Size in kilobytes.
    pub size_kb: f64,
}

impl DiskUsage {
    pub fn combine(self, other: Self) -> Self {
        DiskUsage {
            size_kb: self.size_kb + other.size_kb,
            filepath: self.filepath.into_iter().chain(other.filepath).collect(),
        }
    }
}

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
    PHP,
    Other,
}

lazy_static! {
    static ref NAME_LANGUAGE_MAPPINGS: HashMap<&'static str, Language> = {
        let mut map = HashMap::new();
        for language in Language::iter() {
            for name in language.language_names() {
                map.insert(name, language.clone());
            }
        }
        map
    };
}

impl Language {
    pub fn all_known() -> impl Iterator<Item = Language> {
        Language::iter().filter(|l| l != &Language::Other)
    }

    pub fn language_names(&self) -> Vec<&'static str> {
        match self {
            Language::Rust => vec!["rust"],
            Language::Python => vec!["python"],
            Language::Java => vec!["java"],
            Language::Kotlin => vec!["kotlin"],
            Language::Javascript => vec!["javascript", "javascriptreact"],
            Language::Typescript => vec!["typescript", "typescriptreact"],
            Language::Go => vec!["go"],
            Language::Ruby => vec!["ruby"],
            Language::CSharp => vec!["csharp"],
            Language::C => vec!["c"],
            Language::Cpp => vec!["cpp", "c++"],
            Language::Solidity => vec!["solidity"],
            Language::PHP => vec!["php"],
            Language::Other => vec!["other"],
        }
    }
}

impl From<String> for Language {
    fn from(val: String) -> Self {
        if let Some(lang) = NAME_LANGUAGE_MAPPINGS.get(val.as_str()) {
            lang.clone()
        } else {
            Language::Other
        }
    }
}

#[derive(GraphQLObject, Debug, Clone)]
pub struct ChatCompletionStats {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub user_id: ID,
    pub chats: i32,
}

#[async_trait]
pub trait AnalyticService: Send + Sync {
    /// Generate the completion report for past year, with daily granularity.
    ///
    /// `users` is a list of user IDs. If empty, the report is computed for all users.
    async fn daily_stats_in_past_year(&self, users: Vec<ID>) -> Result<Vec<CompletionStats>>;

    /// Computes the completion report with daily granularity.
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

    /// Generate the chat report for past year, with daily granularity.
    ///
    /// `users` is a list of user IDs. If empty, the report is computed for all users.
    async fn chat_daily_stats_in_past_year(
        &self,
        users: Vec<ID>,
    ) -> Result<Vec<ChatCompletionStats>>;

    /// Computes the chat report with daily granularity.
    ///
    /// 1. [`start`, `end`) define the time range for the report.
    /// 2. `users` is a list of user IDs. If empty, the report is computed for all users.
    async fn chat_daily_stats(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        users: Vec<ID>,
    ) -> Result<Vec<ChatCompletionStats>>;

    async fn disk_usage_stats(&self) -> Result<DiskUsageStats>;
}
