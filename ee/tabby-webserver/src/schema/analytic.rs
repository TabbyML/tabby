use std::fmt::Display;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject, ID};

use crate::schema::Result;

#[derive(GraphQLObject, Debug)]
pub struct CompletionStats {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,

    pub completions: i32,
    pub selects: i32,
}

#[derive(GraphQLEnum, Clone, Debug)]
pub enum Language {
    Rust,
    Python,
    Java,
    Kotlin,
    JavascriptTypescript,
    Go,
    Ruby,
    CSharp,
    C,
    CPlusPlus,
    Solidity,
    Other,
}

impl Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::Rust => write!(f, "rust"),
            Language::Python => write!(f, "python"),
            Language::Java => write!(f, "java"),
            Language::Kotlin => write!(f, "kotlin"),
            Language::JavascriptTypescript => write!(f, "javascript-typescript"),
            Language::Go => write!(f, "go"),
            Language::Ruby => write!(f, "ruby"),
            Language::CSharp => write!(f, "csharp"),
            Language::C => write!(f, "c"),
            Language::CPlusPlus => write!(f, "cpp"),
            Language::Solidity => write!(f, "solidity"),
            Language::Other => write!(f, "other"),
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
