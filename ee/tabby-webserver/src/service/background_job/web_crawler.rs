use std::sync::Arc;

use chrono::Utc;
use futures::StreamExt;
use tabby_inference::Embedding;
use tabby_scheduler::{DocIndexer, WebDocument};
use tabby_crawler::crawl_pipeline;

use super::helper::Job;

pub struct WebCrawlerJob {
    source_id: String,
    url: String,
}

impl Job for WebCrawlerJob {
    const NAME: &'static str = "web_crawler";
}

impl WebCrawlerJob {
    pub fn new(source_id: String, url: String) -> Self {
        Self { source_id, url }
    }

    pub async fn run(self, embedding: Arc<dyn Embedding>) -> tabby_schema::Result<()> {
        logkit::info!("Starting doc index pipeline for {}", self.url);
        let embedding = embedding.clone();
        let mut num_docs = 0;
        let indexer = DocIndexer::new(embedding.clone());

        let mut pipeline = Box::pin(crawl_pipeline(&self.url).await?);
        while let Some(doc) = pipeline.next().await {
            logkit::info!("Fetching {}", doc.url);
            let source_doc = WebDocument {
                source_id: self.source_id.clone(),
                id: doc.url.clone(),
                title: doc.metadata.title.unwrap_or_default(),
                link: doc.url,
                body: doc.markdown,
            };

            num_docs += 1;
            indexer.add(Utc::now(), source_doc).await;
        }
        logkit::info!("Crawled {} documents from '{}'", num_docs, self.url);
        indexer.commit();
        Ok(())
    }
}
