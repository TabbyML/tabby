use async_trait::async_trait;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use utoipa::ToSchema;

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct CodeSearchResponse {
    pub hits: Vec<CodeSearchHit>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CodeSearchHit {
    pub scores: CodeSearchScores,
    pub doc: CodeSearchDocument,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct CodeSearchScores {
    pub combined_rank: i32,
    pub bm25: f32,
    pub embedding: f32,
}

#[derive(Serialize, Deserialize, Debug, Builder, Clone, ToSchema)]
pub struct CodeSearchDocument {
    /// Unique identifier for the file in the repository, stringified SourceFileKey.
    ///
    /// Skipped in API responses.
    #[serde(skip_serializing)]
    pub file_id: String,

    #[serde(skip_serializing)]
    pub chunk_id: String,

    pub body: String,
    pub filepath: String,
    pub git_url: String,
    pub language: String,
    pub start_line: usize,
}

#[derive(Error, Debug)]
pub enum CodeSearchError {
    #[error("index not ready")]
    NotReady,

    #[error(transparent)]
    QueryParserError(#[from] tantivy::query::QueryParserError),

    #[error(transparent)]
    TantivyError(#[from] tantivy::TantivyError),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Deserialize, ToSchema)]
pub struct CodeSearchQuery {
    pub git_url: String,
    pub filepath: Option<String>,
    pub language: String,
    pub content: String,

    #[serde(default)]
    pub scoring_params: CodeSearchScoringParams,
}

#[derive(Deserialize, ToSchema)]
pub struct CodeSearchScoringParams {
    pub min_embedding_score_threshold: f32,
    pub min_bm25_score_threshold: f32,
}

impl Default for CodeSearchScoringParams {
    fn default() -> Self {
        Self {
            min_embedding_score_threshold: 0.75,
            min_bm25_score_threshold: 8.0,
        }
    }
}

#[async_trait]
pub trait CodeSearch: Send + Sync {
    async fn search_in_language(
        &self,
        query: CodeSearchQuery,
        limit: usize,
    ) -> Result<CodeSearchResponse, CodeSearchError>;
}
