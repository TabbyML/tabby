use std::sync::Arc;

use async_stream::stream;
use futures::{stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use tabby_index::public::{
    StructuredDoc, StructuredDocFields, StructuredDocIndexer, StructuredDocIngestedFields,
};
use tabby_inference::Embedding;
use tabby_schema::{ingestion::IngestionService, job::JobService};

use super::{helper::Job, BackgroundJobEvent};

#[derive(Serialize, Deserialize, Clone)]
pub struct IngestionJob;

impl Job for IngestionJob {
    const NAME: &'static str = "ingestion";
}

impl IngestionJob {
    pub async fn cron(
        job: Arc<dyn JobService>,
        ingestion: Arc<dyn IngestionService>,
    ) -> tabby_schema::Result<bool> {
        if !ingestion.should_ingest().await? {
            return Ok(false);
        }

        let _ = job
            .trigger(BackgroundJobEvent::Ingestion.to_command())
            .await;
        Ok(true)
    }

    pub async fn run(
        self,
        ingestion: Arc<dyn IngestionService>,
        embedding: Arc<dyn Embedding>,
    ) -> tabby_schema::Result<()> {
        logkit::info!("Starting ingestion job");

        let docs_stream = match fetch_all_ingested_documents(ingestion).await {
            Ok(s) => s,
            Err(e) => {
                logkit::error!("Failed to fetch ingested documents: {}", e);
                return Err(e);
            }
        };

        let index = StructuredDocIndexer::new(embedding);
        stream! {
            let mut count = 0;
            let mut num_updated = 0;
            for await doc in docs_stream {
                if index.sync(doc).await {
                    num_updated += 1
                }
                count += 1;
                if count % 100 == 0 {
                    logkit::info!("{} ingested documents seen, {} documents updated", count, num_updated);
                };
            }

            logkit::info!("{} ingested documents seen, {} documents updated", count, num_updated);
            index.commit();
        }
        .count()
        .await;

        logkit::info!("Ingestion job completed");

        Ok(())
    }
}

// Ingested documents do not need StructuredDocState
// because they are listed by Pending status, should always be indexed.
async fn fetch_all_ingested_documents(
    ingestion_service: Arc<dyn IngestionService>,
) -> tabby_schema::Result<BoxStream<'static, StructuredDoc>> {
    let s: BoxStream<StructuredDoc> = {
        let stream = stream! {
            let page_size = 10;
            let mut has_more = true;
            let mut after_cursor: Option<String> = None;

            while has_more {
                let ingested_docs = match ingestion_service.list(after_cursor.clone(), None, Some(page_size), None).await {
                    Ok(docs) => docs,
                    Err(e) => {
                        logkit::error!("Failed to fetch ingested documents: {}", e);
                        break;
                    }
                };

                if ingested_docs.is_empty() {
                    break;
                }

                for ingested in ingested_docs.iter() {
                    let doc = StructuredDoc {
                        source_id: ingested.source.clone(),
                        fields: StructuredDocFields::Ingested(StructuredDocIngestedFields {
                            // Add the prefix `/ingested/` to the ID to ensure its uniqueness.
                            id: format!("/ingested/{}/{}", ingested.source, ingested.id),
                            title: ingested.title.clone(),
                            body: ingested.body.clone(),
                            link: ingested.link.clone().unwrap_or_default(),
                        }),
                    };
                    yield doc;
                }

                // If we got fewer documents than the requested page size, we've reached the end
                if ingested_docs.len() < page_size {
                    has_more = false;
                } else {
                    // Get the ID of the last docs to use as the after cursor
                    if let Some(last_doc) = ingested_docs.last() {
                        after_cursor = Some(last_doc.id.to_string());
                    }
                }
            }
        };
        stream.boxed()
    };

    Ok(s)
}
