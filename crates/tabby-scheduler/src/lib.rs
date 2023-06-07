mod index;
mod repository;

use job_scheduler::{Job, JobScheduler};
use tabby_common::config::Config;
use tracing::{error, info};
use anyhow::{Result, anyhow};

pub async fn scheduler(now: bool) -> Result<()> {
    let config = Config::load()?;
    let mut scheduler = JobScheduler::new();

    let job = || {
        info!("Syncing repositories...");
        let ret = repository::sync_repositories(&config);
        if let Err(err) = ret {
            error!("Failed to sync repositories, err: '{}'", err);
            return;
        }

        info!("Indexing repositories...");
        let ret = index::index_repositories(&config);
        if let Err(err) = ret {
            error!("Failed to index repositories, err: '{}'", err);
            return;
        }
    };

    if now {
        job();
    } else {
        // Every 5 hours.
        scheduler.add(Job::new("0 0 1/5 * * * *".parse().unwrap(), job));

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

#[cfg(test)]
mod tests {
    use tabby_common::{
        config::{Config, Repository},
        path::set_tabby_root,
    };
    use temp_testdir::*;
    use tracing_test::traced_test;

    use super::*;

    #[traced_test]
    #[test]
    fn end_to_end() {
        set_tabby_root(TempDir::default().to_path_buf());

        let config = Config {
            repositories: vec![Repository {
                git_url: "https://github.com/TabbyML/interview-questions".to_owned(),
            }],
        };

        repository::sync_repositories(&config).unwrap();
        index::index_repositories(&config).unwrap();
    }
}
