use anyhow::Context;
use async_trait::async_trait;
use humantime::parse_duration;
use tabby_common::api::ingestion::{IngestionRequest, IngestionResponse};
use tabby_db::DbConn;
use tabby_schema::{
    ingestion::{IngestedDocument, IngestionService, IngestionStats},
    CoreError, Result,
};

use crate::service::graphql_pagination_to_filter;

struct IngestionServiceImpl {
    db: DbConn,
}

pub fn create(db: DbConn) -> impl IngestionService {
    IngestionServiceImpl { db }
}

const TTL_DEFAULT_90_DAYS: i64 = 90 * 24 * 60 * 60;

const SOURCE_ID_PREFIX: &str = "ingested:";

#[async_trait]
impl IngestionService for IngestionServiceImpl {
    fn source_name_from_id(&self, source_id: &str) -> String {
        urlencoding::decode(
            source_id
                .strip_prefix(SOURCE_ID_PREFIX)
                .unwrap_or(source_id),
        )
        .unwrap_or_else(|_| source_id.into())
        .to_string()
    }

    fn source_id_from_name(&self, source_name: &str) -> String {
        format!("{}{}", SOURCE_ID_PREFIX, urlencoding::encode(source_name))
    }

    async fn get(&self, source_id: &str, id: &str) -> Result<IngestedDocument> {
        let doc = self
            .db
            .get_ingested_document(source_id, id)
            .await
            .context("Failed to get ingestion")?
            .ok_or_else(|| CoreError::NotFound("Ingested doc not found"))?;

        Ok(doc.into())
    }

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

    async fn list_sources(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<String>> {
        Ok(self
            .db
            .list_ingested_document_sources(limit, offset)
            .await?)
    }

    async fn ingestion(&self, ingestion: IngestionRequest) -> Result<IngestionResponse> {
        let now = chrono::Utc::now();
        let expired_at = if let Some(ttl) = &ingestion.ttl {
            let ttl = parse_duration(ttl)
                .context("Failed to parse TTL")
                .map_err(CoreError::Other)?;
            now.timestamp() + ttl.as_secs() as i64
        } else {
            now.timestamp() + TTL_DEFAULT_90_DAYS
        };

        // url encode the source and id
        let source = self.source_id_from_name(&ingestion.source);
        let id = urlencoding::encode(&ingestion.id);

        self.db
            .upsert_ingested_document(
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
            message: "Ingestion has been accepted and will be processed later.".to_string(),
        })
    }

    async fn delete(&self, source: String, id: String) -> Result<()> {
        let source = self.source_id_from_name(&source);
        let id = urlencoding::encode(&id);

        self.db
            .delete_ingested_document(&source, &id)
            .await
            .context("Failed to delete ingestion")?;
        Ok(())
    }

    async fn delete_by_source_id(&self, source: String) -> Result<()> {
        let source = self.source_id_from_name(&source);
        self.db
            .delete_ingested_document_by_source(&source)
            .await
            .context("Failed to delete ingestion by source")?;
        Ok(())
    }

    async fn stats(&self, sources: Option<Vec<String>>) -> Result<Vec<IngestionStats>> {
        // Assume user provided sources are in the format of source names,
        // we should convert them to source IDs for the database query.
        let source_ids = sources.map(|source_names| {
            source_names
                .into_iter()
                .map(|name| self.source_id_from_name(&name)) // Convert name to ID
                .collect::<Vec<String>>()
        });

        let stats = self.db.list_ingested_document_statuses(source_ids).await?;

        let stats = stats
            .into_iter()
            .map(|stat| {
                let mut stat: IngestionStats = stat.into();

                // Convert source ID back to name for the response
                stat.source = self.source_name_from_id(&stat.source);
                stat
            })
            .collect();

        Ok(stats)
    }

    async fn should_ingest(&self) -> Result<bool> {
        Ok(self.db.count_pending_ingested_documents().await? > 0)
    }

    async fn mark_all_indexed(&self, sourced_ids: Vec<(String, String)>) -> Result<()> {
        for (source, id) in sourced_ids {
            self.db
                .mark_ingested_document_indexed(&source, &id)
                .await
                .context("Failed to mark ingestion as indexed")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ingestion() {
        let db = DbConn::new_in_memory().await.unwrap();
        let ingestion_service = create(db.clone());

        let ingestion = IngestionRequest {
            source: "test_source".to_string(),
            id: "test_id".to_string(),
            title: "Test Title".to_string(),
            body: "Test Body".to_string(),
            link: Some("http://example.com".to_string()),
            ttl: None,
        };

        let response = ingestion_service.ingestion(ingestion).await.unwrap();
        assert_eq!(response.source, "test_source");
        assert_eq!(response.id, "test_id");
    }

    #[tokio::test]
    async fn test_list_sources() {
        let db = DbConn::new_in_memory().await.unwrap();
        let ingestion_service = create(db.clone());

        // Insert some test data
        db.upsert_ingested_document(
            "test_source_1",
            "test_id_1",
            0,
            None,
            "Test Title 1",
            "Test Body 1",
        )
        .await
        .unwrap();

        db.upsert_ingested_document(
            "test_source_1",
            "test_id_2",
            0,
            None,
            "Test Title 2",
            "Test Body 2",
        )
        .await
        .unwrap();

        db.upsert_ingested_document(
            "test_source_2",
            "test_id_2",
            0,
            None,
            "Test Title 2",
            "Test Body 2",
        )
        .await
        .unwrap();

        let sources = ingestion_service.list_sources(None, None).await.unwrap();
        assert_eq!(sources.len(), 2);
        assert!(sources.contains(&"test_source_1".to_string()));
        assert!(sources.contains(&"test_source_2".to_string()));
    }
}
