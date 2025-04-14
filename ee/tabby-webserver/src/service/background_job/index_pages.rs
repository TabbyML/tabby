use std::sync::Arc;

use async_stream::stream;
use futures::{stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use tabby_index::public::{
    StructuredDoc, StructuredDocFields, StructuredDocIndexer, StructuredDocPageFields,
    StructuredDocState,
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
        debug!("Syncing all pages");

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
                logkit::error!("Failed to fetch pages: {}", e);
                return Err(e);
            }
        };

        let index = StructuredDocIndexer::new(embedding);
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
        let stream = stream! {
            let page_size = 10;
            let mut has_more = true;
            let mut after_cursor: Option<String> = None;

            while has_more {
                let pages = match page_service.list(None, after_cursor.clone(), None, Some(page_size), None).await {
                    Ok(pages) => pages,
                    Err(e) => {
                        logkit::error!("Failed to fetch pages: {}", e);
                        break;
                    }
                };

                if pages.is_empty() {
                    break;
                }

                for page in pages.iter() {
                    let state = StructuredDocState {
                        id: page.id.to_string(),
                        updated_at: page.updated_at,
                        deleted: false,
                    };
                    let doc = StructuredDoc {
                        source_id: page_service.source_id(),
                        fields: StructuredDocFields::Page(StructuredDocPageFields {
                            title: page.title.clone().unwrap_or_default(),
                            // must add the prefix `/pages/` to the ID to ensure its uniqueness.
                            // and this can serve as the link to the page.
                            link: format!("/pages/{}", page.id),
                            content: page.content.clone().unwrap_or_default(),
                        }),
                    };
                    yield (state, doc);
                }

                // If we got fewer pages than the requested page size, we've reached the end
                if pages.len() < page_size {
                    has_more = false;
                } else {
                    // Get the ID of the last page to use as the after cursor
                    if let Some(last_page) = pages.last() {
                        after_cursor = Some(last_page.id.to_string());
                    }
                }
            }
        };
        stream.boxed()
    };

    Ok(s)
}
