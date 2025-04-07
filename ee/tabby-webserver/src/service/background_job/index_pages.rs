use std::sync::Arc;

use async_stream::stream;
use futures::{stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use tabby_index::public::{
    StructuredDoc, StructuredDocFields, StructuredDocIndexer, StructuredDocPageFields,
    StructuredDocState, STRUCTURED_DOC_KIND_PAGE,
};
use tabby_inference::Embedding;
use tabby_schema::{job::JobService, page::PageService};
use tracing::debug;

use super::{helper::Job, BackgroundJobEvent};

#[derive(Serialize, Deserialize, Clone)]
pub struct SyncPageIndexJob;

impl Job for SyncPageIndexJob {
    const NAME: &'static str = "page_index_sync";
}

impl SyncPageIndexJob {
    pub async fn cron(job: Arc<dyn JobService>) -> tabby_schema::Result<()> {
        debug!("Syncing all github and gitlab repositories");

        let _ = job
            .trigger(BackgroundJobEvent::SyncPagesIndex.to_command())
            .await;
        Ok(())
    }

    pub async fn run(
        self,
        page: Arc<dyn PageService>,
        embedding: Arc<dyn Embedding>,
    ) -> tabby_schema::Result<()> {
        logkit::info!("Indexing pages");

        let page_stream = match fetch_all_pages(page).await {
            Ok(s) => s,
            Err(e) => {
                logkit::error!("Failed to fetch issues: {}", e);
                return Err(e);
            }
        };

        let index = StructuredDocIndexer::new(embedding, STRUCTURED_DOC_KIND_PAGE);
        stream! {
            let mut count = 0;
            let mut num_updated = 0;
            for await (state, doc) in page_stream {
                if index.presync(&state).await && index.sync(doc).await {
                    num_updated += 1
                }
                count += 1;
                if count % 100 == 0 {
                    logkit::info!("{} pages seen, {} pages updated", count, num_updated);
                };
            }

            logkit::info!("{} pages seen, {} pages updated", count, num_updated);
            index.commit();
        }
        .count()
        .await;

        Ok(())
    }
}

async fn fetch_all_pages(
    page_service: Arc<dyn PageService>,
) -> tabby_schema::Result<BoxStream<'static, (StructuredDocState, StructuredDoc)>> {
    let s: BoxStream<(StructuredDocState, StructuredDoc)> = {
        let pages = page_service.list(None, None, None, None, None).await?;
        let stream = stream! {
            for page in pages {
                let state = StructuredDocState {
                    id: page.id.to_string(),
                    updated_at: page.updated_at,
                    deleted: false,
                };
                let doc = StructuredDoc {
                    source_id: page_service.source_id(),
                    fields: StructuredDocFields::Page(StructuredDocPageFields {
                        title: page.title.unwrap_or_default(),
                        id: page.id.to_string(),
                        content: page.content.unwrap_or_default(),
                    }),
                };
                yield (state, doc);
            }
        };
        stream.boxed()
    };

    Ok(s)
}
