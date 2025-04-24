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
pub struct SyncIngestionIndexJob;

impl Job for SyncIngestionIndexJob {
    const NAME: &'static str = "ingestion_index_sync";
}

impl SyncIngestionIndexJob {
    pub async fn cron(
        job: Arc<dyn JobService>,
        ingestion: Arc<dyn IngestionService>,
    ) -> tabby_schema::Result<bool> {
        if !ingestion.should_ingest().await? {
            return Ok(false);
        }

        let _ = job
            .trigger(BackgroundJobEvent::SyncIngestionIndex.to_command())
            .await;
        Ok(true)
    }

    pub async fn run(
        self,
        ingestion: Arc<dyn IngestionService>,
        embedding: Arc<dyn Embedding>,
    ) -> tabby_schema::Result<()> {
        logkit::info!("Starting ingestion job");

        let mut docs_stream = match fetch_all_ingested_documents(ingestion.clone()).await {
            Ok(s) => s,
            Err(e) => {
                logkit::error!("Failed to fetch ingested documents: {}", e);
                return Err(e);
            }
        }
        .chunks(100); // Commit and update db status after every 100 docs.

        while let Some(docs) = docs_stream.next().await {
            let index = StructuredDocIndexer::new(embedding.clone());
            let ingestion = ingestion.clone();
            stream! {
                let mut ingested_updated = vec![];
                let mut count = 0;
                for ((source, id), doc) in docs.into_iter().inspect(|_| count += 1) {
                    if index.sync(doc).await {
                        ingested_updated.push((source.clone(), id.clone()));
                    }
                }

                index.commit();
                logkit::info!("{} ingested documents seen, {} documents updated", count, ingested_updated.len());
                if let Err(e) = ingestion.mark_all_indexed(ingested_updated).await {
                    logkit::error!("Failed to update ingestion status in database: {}", e);
                }
            }
            .count()
            .await;
        }

        logkit::info!("Ingestion job completed");

        Ok(())
    }
}

// Ingested documents do not need StructuredDocState
// because they are listed by Pending status, should always be indexed.
async fn fetch_all_ingested_documents(
    ingestion_service: Arc<dyn IngestionService>,
) -> tabby_schema::Result<BoxStream<'static, ((String, String), StructuredDoc)>> {
    let s: BoxStream<((String, String), StructuredDoc)> = {
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
                            // Concatenate the source and id to ensure uniqueness,
                            // with the source having an `ingested:` prefix to guarantee it is distinct.
                            id: format!("{}/{}", ingested.source, ingested.id),
                            title: ingested.title.clone(),
                            body: ingested.body.clone(),
                            link: ingested.link.clone(),
                        }),
                    };
                    yield ((ingested.source.clone(), ingested.id.clone()), doc);
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
