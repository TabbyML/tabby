use crate::Result;
use async_trait::async_trait;
use tabby_common::api::ingestion::{IngestionRequest, IngestionResponse};

#[async_trait]
pub trait IngestionService: Send + Sync {
    async fn ingestion(&self, ingestion: IngestionRequest) -> Result<IngestionResponse>;
}
