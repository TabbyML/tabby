mod dataset;
mod index;
mod repository;
mod utils;

use anyhow::Result;
use job_scheduler::{Job, JobScheduler};
use tabby_common::config::Config;
use tracing::{error, info};

pub async fn scheduler(now: bool) -> Result<()> {
    let config = Config::load()?;
    let mut scheduler = JobScheduler::new();

    let job1 = || {
        dataset::sync_repository(&config);
    };

    let job2 = || {
        index::index_repository(&config);
    };

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

#[cfg(feature = "ee")]
pub fn job_sync() {
    let Ok(config) = Config::load() else {
        error!("Scheduler job failed to load config");
        return;
    };
    dataset::sync_repository(&config)
}

#[cfg(feature = "ee")]
pub fn job_index() {
    let Ok(config) = Config::load() else {
        error!("Scheduler job failed to load config");
        return;
    };
    index::index_repository(&config)
}
