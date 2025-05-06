use std::sync::Arc;

use tabby_index::public::{
    run_index_garbage_collection, CodeIndexer, StructuredDocGarbageCollector,
};
use tabby_schema::{
    context::ContextService, ingestion::IngestionService, repository::RepositoryService,
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
        context: Arc<dyn ContextService>,
        ingestion: Arc<dyn IngestionService>,
    ) -> tabby_schema::Result<()> {
        let mut failed = false;
        // Run garbage collection on the index
        let sources = context
            .read(None)
            .await?
            .sources
            .into_iter()
            .map(|x| x.source_id())
            .collect::<Vec<_>>();
        if let Err(e) = run_index_garbage_collection(sources) {
            failed = true;
            logkit::warn!("Failed to run index garbage collection: {}", e);
        }

        // Run garbage collection on the code repositories (cloned directories)
        let repositories = repository.list_all_code_repository().await?;
        let mut code = CodeIndexer::default();
        code.garbage_collection(&repositories).await;

        let should_keep_structured_doc = |source_id: String, id: String| {
            let ingestion = ingestion.clone();
            async move { ingestion.get(&source_id, &id).await.is_ok() }
        };

        let structured_doc_garbage_collector = StructuredDocGarbageCollector::default();
        if let Err(e) = structured_doc_garbage_collector
            .run(should_keep_structured_doc)
            .await
        {
            failed = true;
            logkit::warn!("Failed to run structured doc garbage collection: {}", e);
        }

        if failed {
            logkit::warn!("Index garbage collection job failed");
            Err(tabby_schema::CoreError::Other(anyhow::anyhow!(
                "Index garbage collection job failed"
            )))
        } else {
            logkit::info!("Index garbage collection job completed successfully");
            Ok(())
        }
    }
}
