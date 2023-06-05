mod repository;

use std::time::Duration;
use tracing::info;
use job_scheduler::{JobScheduler, Job};

pub fn scheduler() {
    let mut scheduler = JobScheduler::new();

        // Every 5 hours.
    scheduler.add(Job::new("* * 1/5 * * *".parse().unwrap(), || {
        info!("Syncing repositories...");
        repository::sync_repositories();
    }));

    info!("Scheduler activated...");
    loop {
        info!("Checking for jobs in queue...");
        scheduler.tick();
        std::thread::sleep(Duration::from_secs(10));
    }
}
