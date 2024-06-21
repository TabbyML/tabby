use std::sync::Arc;

use tabby_inference::Embedding;

use super::helper::Job;

pub struct WebCrawlerJob {
    url: String,
}

impl Job for WebCrawlerJob {
    const NAME: &'static str = "web";
}

impl WebCrawlerJob {
    pub fn new(url: String) -> Self {
        Self { url }
    }

    pub async fn run(self, embedding: Arc<dyn Embedding>) {
        tabby_scheduler::crawl_index_docs(&[self.url], embedding).await;
    }
}
