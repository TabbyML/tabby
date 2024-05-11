//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.
mod cache;
mod code;
mod dataset;
mod index;
mod repository;

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
    let mut manager = RepositoryManager::default();
    for repository in repositories {
        manager.refresh(repository);
    }

    create_dataset(repositories);
    manager.garbage_collection();
}

fn create_dataset(repositories: &[RepositoryConfig]) {
    let mut cache = cache::CacheStore::new(tabby_common::path::cache_dir());
    dataset::create_dataset(&mut cache, repositories);
}

#[derive(Default)]
pub struct RepositoryManager {
    is_dirty: bool,
}

impl RepositoryManager {
    pub fn refresh(&mut self, repository: &RepositoryConfig) {
        self.is_dirty = true;

        info!("Refreshing repository: {}", repository.git_url);
        repository::sync_repository(repository);

        let mut cache = cache::CacheStore::new(tabby_common::path::cache_dir());
        index::index_repository(&mut cache, repository);
    }

    pub fn garbage_collection(&mut self) {
        self.is_dirty = false;
        let mut cache = cache::CacheStore::new(tabby_common::path::cache_dir());
        cache.garbage_collection_for_source_files();
        index::garbage_collection(&mut cache);
    }
}

impl Drop for RepositoryManager {
    fn drop(&mut self) {
        if self.is_dirty {
            warn!("Garbage collection was expected to be invoked at least once but was not.")
        }
    }
}

pub mod crawl;
