//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.

pub mod crawl;

mod code;
pub use code::CodeIndex;

mod doc;
pub use doc::DocIndex;

use std::sync::Arc;

use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{info, warn};

pub async fn scheduler<T: RepositoryAccess + 'static>(now: bool, access: T) {
    if now {
        let repositories = access
            .list_repositories()
            .await
            .expect("Must be able to retrieve repositories for sync");
        scheduler_pipeline(&repositories);
    } else {
        let access = Arc::new(access);
        let scheduler = JobScheduler::new()
            .await
            .expect("Failed to create scheduler");
        let scheduler_mutex = Arc::new(tokio::sync::Mutex::new(()));

        // Every 10 minutes
        scheduler
            .add(
                Job::new_async("0 1/10 * * * *", move |_, _| {
                    let access = access.clone();
                    let scheduler_mutex = scheduler_mutex.clone();
                    Box::pin(async move {
                        let Ok(_guard) = scheduler_mutex.try_lock() else {
                            warn!("Scheduler job overlapped, skipping...");
                            return;
                        };

                        let repositories = access
                            .list_repositories()
                            .await
                            .expect("Must be able to retrieve repositories for sync");

                        scheduler_pipeline(&repositories);
                    })
                })
                .expect("Failed to create job"),
            )
            .await
            .expect("Failed to add job");

        info!("Scheduler activated...");
        scheduler.start().await.expect("Failed to start scheduler");

        // Sleep 10 years (indefinitely)
        tokio::time::sleep(tokio::time::Duration::from_secs(3600 * 24 * 365 * 10)).await;
    }
}

fn scheduler_pipeline(repositories: &[RepositoryConfig]) {
    let mut code = CodeIndex::default();
    for repository in repositories {
        code.refresh(repository);
    }

    code.garbage_collection();
}

mod tantivy_utils {
    use std::{fs, path::Path};

    use tabby_common::index::register_tokenizers;
    use tantivy::{directory::MmapDirectory, schema::Schema, Index};
    use tracing::{debug, warn};

    pub fn open_or_create_index(code: &Schema, path: &Path) -> Index {
        let index = match open_or_create_index_impl(code, path) {
            Ok(index) => index,
            Err(err) => {
                warn!(
                    "Failed to open index repositories: {}, removing index directory '{}'...",
                    err,
                    path.display()
                );
                fs::remove_dir_all(path).expect("Failed to remove index directory");

                debug!("Reopening index repositories...");
                open_or_create_index_impl(code, path).expect("Failed to open index")
            }
        };
        register_tokenizers(&index);
        index
    }

    fn open_or_create_index_impl(code: &Schema, path: &Path) -> tantivy::Result<Index> {
        fs::create_dir_all(path).expect("Failed to create index directory");
        let directory = MmapDirectory::open(path).expect("Failed to open index directory");
        Index::open_or_create(directory, code.clone())
    }
}
