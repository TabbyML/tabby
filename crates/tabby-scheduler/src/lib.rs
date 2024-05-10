//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.
mod cache;
mod code;
mod dataset;
mod index;
mod repository;
mod utils;

use std::sync::Arc;

use cache::CacheStore;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{info, warn};

pub async fn scheduler<T: RepositoryAccess + 'static>(now: bool, access: T) {
    if now {
        let mut cache = cache::CacheStore::new(tabby_common::path::cache_dir());
        let repositories = access
            .list_repositories()
            .await
            .expect("Must be able to retrieve repositories for sync");
        job_sync(&mut cache, &repositories);
        job_index(&mut cache, &repositories);

        cache.garbage_collection_for_source_files();
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

                        let mut cache = cache::CacheStore::new(tabby_common::path::cache_dir());

                        let repositories = access
                            .list_repositories()
                            .await
                            .expect("Must be able to retrieve repositories for sync");

                        job_sync(&mut cache, &repositories);
                        job_index(&mut cache, &repositories);
                        cache.garbage_collection_for_source_files();
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

fn job_index(cache: &mut CacheStore, repositories: &[RepositoryConfig]) {
    println!("Indexing repositories...");
    index::index_repositories(cache, repositories);
}

fn job_sync(cache: &mut CacheStore, repositories: &[RepositoryConfig]) {
    println!("Syncing {} repositories...", repositories.len());
    repository::sync_repositories(repositories);

    println!("Creating dataset...");
    dataset::create_dataset(cache, repositories);
}
