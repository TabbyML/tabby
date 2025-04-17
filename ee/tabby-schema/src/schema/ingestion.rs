use crate::Result;
use async_trait::async_trait;
use tabby_common::api::ingestion::{IngestionRequest, IngestionResponse};

pub struct IngestedDocument {
    pub id: String,
    pub source: String,
    pub link: Option<String>,
    pub title: String,
    pub body: String,
    pub status: IngestionStatus,
}

pub enum IngestionStatus {
    Pending,
    Indexed,
    Failed,
}

#[async_trait]
pub trait IngestionService: Send + Sync {
    async fn list(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<IngestedDocument>>;

    async fn ingestion(&self, ingestion: IngestionRequest) -> Result<IngestionResponse>;

    async fn should_ingest(&self) -> Result<bool>;
}
