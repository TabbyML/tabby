//! Responsible for scheduling all of the background jobs for tabby.
//! Includes syncing respositories and updating indices.
mod dataset;
mod index;
mod repository;
mod utils;

use std::sync::Arc;

use anyhow::Result;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{error, info, warn};

pub async fn scheduler<T: RepositoryAccess + 'static>(now: bool, access: T) -> Result<()> {
    if now {
        let repositories = access.list_repositories().await?;
        job_sync(&repositories)?;
        job_index(&repositories)?;
    } else {
        let access = Arc::new(access);
        let scheduler = JobScheduler::new().await?;
        let scheduler_mutex = Arc::new(tokio::sync::Mutex::new(()));

        // Every 10 minutes
        scheduler
            .add(Job::new_async("0 1/10 * * * *", move |_, _| {
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
                    if let Err(e) = job_sync(&repositories) {
                        error!("{e}");
                    }
                    if let Err(e) = job_index(&repositories) {
                        error!("{e}")
                    }
                })
            })?)
            .await?;

        info!("Scheduler activated...");
        scheduler.start().await?;

        // Sleep 10 years (indefinitely)
        tokio::time::sleep(tokio::time::Duration::from_secs(3600 * 24 * 365 * 10)).await;
    }

    Ok(())
}

fn job_index(repositories: &[RepositoryConfig]) -> Result<()> {
    println!("Indexing repositories...");
    let ret = index::index_repositories(repositories);
    if let Err(err) = ret {
        return Err(err.context("Failed to index repositories"));
    }
    Ok(())
}

fn job_sync(repositories: &[RepositoryConfig]) -> Result<()> {
    println!("Syncing {} repositories...", repositories.len());
    let ret = repository::sync_repositories(repositories);
    if let Err(err) = ret {
        return Err(err.context("Failed to sync repositories"));
    }

    println!("Building dataset...");
    let ret = dataset::create_dataset(repositories);
    if let Err(err) = ret {
        return Err(err.context("Failed to build dataset"));
    }
    Ok(())
}
