use std::sync::Arc;

use chrono::{DateTime, Utc};
use tabby_inference::Embedding;
use tabby_schema::web_crawler::WebCrawlerService;
use tabby_schema::Result;
use tokio::sync::mpsc::UnboundedSender;

use super::helper::Job;
use super::BackgroundJobEvent;

pub struct WebCrawlerJob {
    url: String,
}

impl Job for WebCrawlerJob {
    const NAME: &'static str = "web_crawler";
}

impl WebCrawlerJob {
    pub fn new(url: String) -> Self {
        Self { url }
    }

    pub async fn run(self, embedding: Arc<dyn Embedding>) {
        tabby_scheduler::crawl_index_docs(&[self.url], embedding).await;
    }

    pub async fn cron(
        _now: DateTime<Utc>,
        sender: UnboundedSender<BackgroundJobEvent>,
        web_crawler_service: Arc<dyn WebCrawlerService>,
    ) -> Result<()> {
        for url in web_crawler_service
            .list_web_crawler_urls(None, None, None, None)
            .await?
        {
            sender
                .send(BackgroundJobEvent::WebCrawler(url.url))
                .expect("Failed to enqueue web crawler job");
        }
        Ok(())
    }
}
