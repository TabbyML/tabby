use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use utoipa::ToSchema;

#[derive(Default, Serialize, Deserialize, Debug, ToSchema)]
pub struct DocSearchResponse {
    pub hits: Vec<DocSearchHit>,
}

#[derive(Serialize, Deserialize, Debug, ToSchema)]
pub struct DocSearchHit {
    pub score: f32,
    pub doc: DocSearchDocument,
}

#[derive(Serialize, Deserialize, Debug, ToSchema)]
pub struct DocSearchDocument {
    pub title: String,
    pub link: String,
    pub snippet: String,
}

#[derive(Error, Debug)]
pub enum DocSearchError {
    #[error("index not ready")]
    NotReady,

    #[error(transparent)]
    QueryParserError(#[from] tantivy::query::QueryParserError),

    #[error(transparent)]
    TantivyError(#[from] tantivy::TantivyError),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[async_trait]
pub trait DocSearch: Send + Sync {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<DocSearchResponse, DocSearchError>;
}
