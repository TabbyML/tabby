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
        job_sync(&repositories);
        job_index(&repositories);
    } else {
        let access = Arc::new(access);
        let scheduler = JobScheduler::new().await?;
        // Every 5 minutes.
        let access_clone = access.clone();
        scheduler
            .add(Job::new_async("0 1/5 * * * * *", move |_, _| {
                let access = access_clone.clone();
                Box::pin(async move {
                    match access.list_repositories().await {
                        Ok(repositories) => job_sync(&repositories),
                        Err(err) => warn!("Failed to list_repositories: {}", err),
                    }
                })
            })?)
            .await?;

        // Every 5 hours.
        let access_clone = access.clone();
        scheduler
            .add(Job::new_async("0 0 1/5 * * * *", move |_, _| {
                let access = access_clone.clone();
                Box::pin(async move {
                    match access.list_repositories().await {
                        Ok(repositories) => job_index(&repositories),
                        Err(err) => warn!("Failed to list_repositories: {}", err),
                    }
                })
            })?)
            .await?;

        info!("Scheduler activated...");
        scheduler.start().await?;
    }

    Ok(())
}

fn job_index(repositories: &[RepositoryConfig]) {
    println!("Indexing repositories...");
    let ret = index::index_repositories(repositories);
    if let Err(err) = ret {
        error!("Failed to index repositories, err: '{}'", err);
    }
    println!();
}

fn job_sync(repositories: &[RepositoryConfig]) {
    println!("Syncing repositories...");
    let ret = repository::sync_repositories(repositories);
    if let Err(err) = ret {
        error!("Failed to sync repositories, err: '{}'", err);
        return;
    }

    println!("Building dataset...");
    let ret = dataset::create_dataset(repositories);
    if let Err(err) = ret {
        error!("Failed to build dataset, err: '{}'", err);
    }
    println!();
}
