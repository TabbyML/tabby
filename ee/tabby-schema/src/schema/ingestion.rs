use async_trait::async_trait;
use tabby_common::api::ingestion::{IngestionRequest, IngestionResponse};

use crate::Result;

pub struct IngestedDocument {
    pub id: String,
    pub source: String,
    pub link: Option<String>,
    pub title: String,
    pub body: String,
    pub status: IngestedDocStatus,
}

pub enum IngestedDocStatus {
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

    async fn list_sources(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<String>>;

    async fn ingestion(&self, ingestion: IngestionRequest) -> Result<IngestionResponse>;

    async fn should_ingest(&self) -> Result<bool>;
    async fn mark_all_indexed(&self, sourced_ids: Vec<(String, String)>) -> Result<()>;

    fn source_name_from_id(&self, source_id: &str) -> String;
    fn source_id_from_name(&self, source_name: &str) -> String;
}
