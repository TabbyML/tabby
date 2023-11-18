use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use utoipa::ToSchema;

#[derive(Default, Serialize, Deserialize, Debug, ToSchema)]
pub struct SearchResponse {
    pub num_hits: usize,
    pub hits: Vec<Hit>,
}

#[derive(Serialize, Deserialize, Debug, ToSchema)]
pub struct Hit {
    pub score: f32,
    pub doc: HitDocument,
    pub id: u32,
}

#[derive(Serialize, Deserialize, Debug, ToSchema)]
pub struct HitDocument {
    pub body: String,
    pub filepath: String,
    pub git_url: String,
    pub kind: String,
    pub language: String,
    pub name: String,
}

#[derive(Error, Debug)]
pub enum CodeSearchError {
    #[error("index not ready")]
    NotReady,

    #[error(transparent)]
    QueryParserError(#[from] tantivy::query::QueryParserError),

    #[error(transparent)]
    TantivyError(#[from] tantivy::TantivyError),
}

#[async_trait]
pub trait CodeSearch: Send + Sync {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError>;

    async fn search_in_language(
        &self,
        language: &str,
        tokens: &[String],
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError>;
}
