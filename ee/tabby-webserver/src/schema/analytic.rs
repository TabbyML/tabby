use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject, ID};
use strum::{EnumIter, IntoEnumIterator};

use crate::schema::Result;

#[derive(GraphQLObject, Debug)]
pub struct CompletionStats {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,

    pub completions: i32,
    pub selects: i32,
}

#[derive(GraphQLEnum, Clone, Debug, Eq, PartialEq, EnumIter)]
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
    CPP,
    Solidity,
    Other,
}

impl Language {
    pub fn all_known() -> impl Iterator<Item = Language> {
        Language::iter().filter(|l| l != &Language::Other)
    }

    pub fn to_strings(&self) -> impl IntoIterator<Item = String> {
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
            Language::CPP => vec!["cpp"],
            Language::Solidity => vec!["solidity"],
            Language::Other => vec!["other"],
        }
        .into_iter()
        .map(|s| s.to_string())
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
