mod dataset;
mod index;
mod repository;
mod utils;

use anyhow::Result;
use job_scheduler::{Job, JobScheduler};
use tabby_common::config::Config;
use tracing::{error, info};

pub async fn scheduler(now: bool, config: &Config) -> Result<()> {
    let mut scheduler = JobScheduler::new();

    let job1 = || job_sync(config);

    let job2 = || job_index(config);

    if now {
        job1();
        job2();
    } else {
        // Every 5 minutes.
        scheduler.add(Job::new("0 1/5 * * * * *".parse().unwrap(), job1));

        // Every 5 hours.
        scheduler.add(Job::new("0 0 1/5 * * * *".parse().unwrap(), job2));

        info!("Scheduler activated...");
        loop {
            scheduler.tick();
            let duration = scheduler.time_till_next_job();
            info!("Sleep {:?} for next job ...", duration);
            std::thread::sleep(duration);
        }
    }

    Ok(())
}

pub fn job_index(config: &Config) {
    println!("Indexing repositories...");
    let ret = index::index_repositories(config);
    if let Err(err) = ret {
        error!("Failed to index repositories, err: '{}'", err);
    }
    println!();
}

pub fn job_sync(config: &Config) {
    println!("Syncing repositories...");
    let ret = repository::sync_repositories(config);
    if let Err(err) = ret {
        error!("Failed to sync repositories, err: '{}'", err);
        return;
    }

    println!("Building dataset...");
    let ret = dataset::create_dataset(config);
    if let Err(err) = ret {
        error!("Failed to build dataset, err: '{}'", err);
    }
    println!();
}
