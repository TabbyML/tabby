use std::sync::Arc;

use tabby_common::index::corpus;
use tabby_scheduler::run_index_garbage_collection;
use tabby_schema::{repository::RepositoryService, web_crawler::WebCrawlerService};

use super::helper::Job;

pub struct IndexGarbageCollection;

impl Job for IndexGarbageCollection {
    const NAME: &'static str = "index_garbage_collection";
}

impl IndexGarbageCollection {
    pub async fn run(
        self,
        repository: Arc<dyn RepositoryService>,
        web_crawler: Arc<dyn WebCrawlerService>,
    ) -> tabby_schema::Result<()> {
        let mut sources: Vec<_> = repository.list_all_sources().await?.into_iter().collect();

        sources.extend(
            web_crawler
                .list_web_crawler_urls(None, None, None, None)
                .await?
                .into_iter()
                .map(|url| (corpus::WEB.into(), url.source_id())),
        );

        run_index_garbage_collection(sources).map_err(tabby_schema::CoreError::Other)
    }
}
