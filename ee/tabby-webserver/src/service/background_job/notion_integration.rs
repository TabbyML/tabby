

use std::{sync::Arc, time::Duration};
use anyhow::{Context};
use futures::TryFutureExt;
use serde::{Deserialize, Serialize};

use tabby_db::notion_documents::NotionDocumentType;
use tabby_index::public::{DocIndexer, WebDocument};
use tabby_inference::Embedding;
use tabby_schema::CoreError;

use super::notion::notion_util;
use super::helper::Job;

const CRAWLER_TIMEOUT_SECS: u64 = 7200;

#[derive(Debug, Serialize, Deserialize)]
pub struct NotionJob {
    pub source_id: String,
    pub access_token: String,
    pub integration_id: String,
    pub integration_type: NotionDocumentType,
}

impl Job for NotionJob {
    const NAME: &'static str = "";
}

impl NotionJob {

    pub async fn run_impl(self, embedding: Arc<dyn Embedding>) -> tabby_schema::Result<()> {
        
        logkit::info!("Starting notion database index pipeline for {}", self.integration_id);
        let embedding = embedding.clone();
        let mut num_docs = 0;
        let indexer = DocIndexer::new(embedding.clone());
        let pages = notion_util::fetch_all_pages(
            self.access_token.as_str(), 
            self.integration_id.as_str()).
            await.map_err(|e|CoreError::Other(e.into()))?;
        for page in pages {

            let source_doc = WebDocument {
                source_id: self.source_id.clone(),
                id: page.id(),
                title: page.title(),
                link: page.url(),
                body: page.plain_text(),
            };
            num_docs += 1;
            debug!("fetched page: {}, {}", &source_doc.title,&source_doc.body);
            indexer.add(page.last_edited_time(), source_doc).await;
        }
        logkit::info!("fetched {} pages from notion:'{}'", num_docs, self.integration_id);
        indexer.commit();
        return Ok(());
    }

    pub async fn run(self, embedding: Arc<dyn Embedding>) -> tabby_schema::Result<()> {
        let notion_id = self.integration_id.clone();
        if tokio::time::timeout(
            Duration::from_secs(CRAWLER_TIMEOUT_SECS),
            self.run_impl(embedding),
        )
        .await
        .is_err()
        {
            logkit::warn!(
                "fetched for notion: {} timeout after {} seconds",
                notion_id,
                CRAWLER_TIMEOUT_SECS
            );
        }
        Ok(())
    }
    
}

