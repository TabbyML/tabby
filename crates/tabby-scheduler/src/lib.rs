mod index;
mod repository;

use job_scheduler::{Job, JobScheduler};
use tabby_common::config::Config;
use tracing::{error, info};

pub fn scheduler(now: bool) {
    let config = Config::load();
    if config.is_err() {
        error!("Please create config.toml before using scheduler");
        return;
    }

    let config = config.unwrap();
    let mut scheduler = JobScheduler::new();

    let job = || {
        info!("Syncing repositories...");
        repository::sync_repositories(&config);

        info!("Indexing repositories...");
        index::index_repositories(&config);
    };

    if now {
        job()
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

        repository::sync_repositories(&config);
        index::index_repositories(&config);
    }
}
