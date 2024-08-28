use std::sync::Arc;

use tabby_index::public::{run_index_garbage_collection, CodeIndexer};
use tabby_schema::{
    repository::RepositoryService, web_crawler::WebCrawlerService,
    web_documents::WebDocumentService,
};

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
        web_document: Arc<dyn WebDocumentService>,
    ) -> tabby_schema::Result<()> {
        let repositories = repository.list_all_code_repository().await?;
        let mut sources: Vec<_> = repositories
            .iter()
            .map(|repo| repo.source_id.clone())
            .collect();

        sources.extend(
            web_crawler
                .list_web_crawler_urls(None, None, None, None)
                .await?
                .into_iter()
                .map(|url| url.source_id()),
        );

        sources.extend(
            web_document
                .list_custom_web_documents(None, None, None, None, None)
                .await?
                .into_iter()
                .map(|doc| doc.source_id()),
        );

        sources.extend(
            web_document
                .list_preset_web_documents(None, None, None, None, None, Some(true))
                .await?
                .into_iter()
                .map(|doc| doc.source_id()),
        );

        // Run garbage collection on the index
        run_index_garbage_collection(sources)?;

        // Run garbage collection on the code repositories (cloned directories)
        let mut code = CodeIndexer::default();
        code.garbage_collection(&repositories).await;

        Ok(())
    }
}
