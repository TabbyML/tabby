use async_trait::async_trait;
use serde::Serialize;
use thiserror::Error;
use utoipa::ToSchema;

#[derive(Serialize, ToSchema)]
pub struct SearchResponse {
    pub num_hits: usize,
    pub hits: Vec<Hit>,
}

#[derive(Serialize, ToSchema)]
pub struct Hit {
    pub score: f32,
    pub doc: HitDocument,
    pub id: u32,
}

#[derive(Serialize, ToSchema)]
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

    #[error("{0}")]
    QueryParserError(#[from] tantivy::query::QueryParserError),

    #[error("{0}")]
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

    async fn search_with_query(
        &self,
        q: &dyn tantivy::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError>;
}
