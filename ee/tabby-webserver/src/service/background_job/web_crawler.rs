use std::sync::Arc;

use juniper::ID;
use tabby_inference::Embedding;

use crate::service::web_crawler::format_website_source;

use super::{
    cprintln,
    helper::{Job, JobLogger},
};

pub struct WebCrawlerJob {
    url: String,
    id: ID,
}

impl Job for WebCrawlerJob {
    const NAME: &'static str = "web";
}

impl WebCrawlerJob {
    pub fn new(url: String, id: ID) -> Self {
        Self { url, id }
    }

    pub async fn run(
        self,
        job_logger: JobLogger,
        embedding: Arc<dyn Embedding>,
    ) -> tabby_schema::Result<()> {
        let source = format_website_source(self.id);
        tabby_scheduler::crawl_index_docs(&[self.url], embedding, source, move |url| {
            let job_logger = job_logger.clone();
            async move { cprintln!(job_logger, "Fetching {url}") }
        })
        .await?;
        Ok(())
    }
}
