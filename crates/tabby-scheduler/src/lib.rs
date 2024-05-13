//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.
mod code;
pub use code::CodeIndex;

pub mod crawl;

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
