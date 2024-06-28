use std::sync::Arc;

use tabby_inference::Embedding;

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
        tabby_scheduler::crawl_index_docs(&self.source_id, &self.url, embedding).await?;
        Ok(())
    }
}
