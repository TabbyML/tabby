use anyhow::Context;
use async_trait::async_trait;
use humantime::parse_duration;
use tabby_common::api::ingestion::{IngestionRequest, IngestionResponse, IngestionStatus};
use tabby_db::DbConn;
use tabby_schema::{
    ingestion::{IngestedDocument, IngestionService},
    CoreError, Result,
};
use urlencoding;

use crate::service::graphql_pagination_to_filter;

struct IngestionServiceImpl {
    db: DbConn,
}

pub fn create(db: DbConn) -> impl IngestionService {
    IngestionServiceImpl { db }
}

const TTL_DEFAULT_90_DAYS: i64 = 90 * 24 * 60 * 60;

#[async_trait]
impl IngestionService for IngestionServiceImpl {
    async fn list(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<IngestedDocument>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;

        let docs = self
            .db
            .list_ingested_documents(limit, skip_id, backwards)
            .await?;

        Ok(docs.into_iter().map(Into::into).collect())
    }

    async fn ingestion(&self, ingestion: IngestionRequest) -> Result<IngestionResponse> {
        let now = chrono::Utc::now();
        let expired_at = if let Some(ttl) = &ingestion.ttl {
            let ttl = parse_duration(ttl)
                .context("Failed to parse TTL")
                .map_err(|e| CoreError::Other(e))?;
            now.timestamp() + ttl.as_secs() as i64
        } else {
            now.timestamp() + TTL_DEFAULT_90_DAYS
        };

        // url encode the source and id
        let source = urlencoding::encode(&ingestion.source);
        let id = urlencoding::encode(&ingestion.id);

        self.db
            .insert_ingested_document(
                &source,
                &id,
                expired_at,
                ingestion.link,
                &ingestion.title,
                &ingestion.body,
            )
            .await
            .context("Failed to create ingestion")?;

        Ok(IngestionResponse {
            id: ingestion.id,
            source: ingestion.source,
            status: IngestionStatus::Pending,
            message: "Ingestion has been accepted and will be processed later.".to_string(),
        })
    }

    async fn should_ingest(&self) -> Result<bool> {
        Ok(self.db.count_pending_ingested_documents().await? > 0)
    }
}
