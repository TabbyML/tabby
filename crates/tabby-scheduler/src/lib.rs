mod dataset;
mod index;
mod repository;
mod utils;

use anyhow::Result;
pub use index::open_or_create_index;
use job_scheduler::{Job, JobScheduler};
use tabby_common::config::Config;
use tracing::{error, info};

pub async fn scheduler(now: bool) -> Result<()> {
    let config = Config::load()?;
    let mut scheduler = JobScheduler::new();

    let job1 = || {
        println!("Syncing repositories...");
        let ret = repository::sync_repositories(&config);
        if let Err(err) = ret {
            error!("Failed to sync repositories, err: '{}'", err);
            return;
        }

        println!("Building dataset...");
        let ret = dataset::create_dataset(&config);
        if let Err(err) = ret {
            error!("Failed to build dataset, err: '{}'", err);
        }
        println!();
    };

    let job2 = || {
        println!("Indexing repositories...");
        let ret = index::index_repositories(&config);
        if let Err(err) = ret {
            error!("Failed to index repositories, err: '{}'", err);
        }
        println!()
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
