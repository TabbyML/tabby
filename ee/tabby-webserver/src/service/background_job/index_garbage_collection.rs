use std::sync::Arc;

use tabby_index::public::{run_index_garbage_collection, CodeIndexer};
use tabby_schema::{context::ContextService, repository::RepositoryService};

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
    ) -> tabby_schema::Result<()> {
        // Run garbage collection on the index
        let sources = context
            .read(None)
            .await?
            .sources
            .into_iter()
            .map(|x| x.source_id())
            .collect::<Vec<_>>();
        run_index_garbage_collection(sources)?;

        // Run garbage collection on the code repositories (cloned directories)
        let repositories = repository.list_all_code_repository().await?;
        let mut code = CodeIndexer::default();
        code.garbage_collection(&repositories).await;

        Ok(())
    }
}
