use std::sync::Arc;

use serde::Serialize;
use tabby_index::public::{run_index_garbage_collection, CodeIndexer};
use tabby_schema::{context::ContextService, repository::RepositoryService};

use super::helper::{Job, JobLogger};

#[derive(Serialize)]
pub struct IndexGarbageCollection;

impl Job for IndexGarbageCollection {
    const NAME: &'static str = "index_garbage_collection";
}

impl IndexGarbageCollection {
    pub async fn run(
        self,
        repository: Arc<dyn RepositoryService>,
        context: Arc<dyn ContextService>,
        db: tabby_db::DbConn,
        job_id: i64,
    ) -> tabby_schema::Result<()> {
        let logger = JobLogger::new(db.clone(), job_id);

        // Run garbage collection on the index
        let sources = match context.read(None).await {
            Ok(sources) => sources,
            Err(err) => {
                logkit::warn!(exit_code = -1; "Failed to list sources: {}", err);
                logger.finalize().await;
                return Err(err);
            }
        };
        let sources = sources
            .sources
            .into_iter()
            .map(|x| x.source_id())
            .collect::<Vec<_>>();

        if let Err(e) = run_index_garbage_collection(sources) {
            logkit::warn!(exit_code = -1; "Failed to run index garbage collection: {}", e);
            logger.finalize().await;
            return Err(e.into());
        }

        // Run garbage collection on the code repositories (cloned directories)
        let repositories = match repository.list_all_code_repository().await {
            Ok(repos) => repos,
            Err(err) => {
                logkit::warn!(exit_code = -1; "Failed to list repositories: {}", err);
                logger.finalize().await;
                return Err(err);
            }
        };
        let mut code = CodeIndexer::default();
        code.garbage_collection(&repositories).await;

        logger.finalize().await;
        Ok(())
    }
}
