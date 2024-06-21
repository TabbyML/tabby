use std::sync::Arc;

use tabby_inference::Embedding;

use super::{
    cprintln,
    helper::{Job, JobLogger},
};

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

    pub async fn run(
        self,
        job_logger: JobLogger,
        embedding: Arc<dyn Embedding>,
    ) -> tabby_schema::Result<()> {
        tabby_scheduler::crawl_index_docs(&[self.url], embedding, move |url| {
            let job_logger = job_logger.clone();
            async move { cprintln!(job_logger, "Fetching {url}") }
        })
        .await?;
        Ok(())
    }
}
