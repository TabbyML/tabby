mod index;
mod repository;

use std::time::Duration;

use job_scheduler::{Job, JobScheduler};
use tracing::info;

pub fn scheduler(run: bool) {
    let config = Config::load();

    let mut scheduler = JobScheduler::new();

    let job = || {
        info!("Syncing repositories...");
        repository::sync_repositories(&config);

        info!("Indexing repositories...");
        index::index_repositories(&config);
    };

    if run {
        job()
    } else {
        // Every 5 hours.
        scheduler.add(Job::new("* * 1/5 * * *".parse().unwrap(), job));

        info!("Scheduler activated...");
        loop {
            info!("Checking for jobs in queue...");
            scheduler.tick();
            std::thread::sleep(Duration::from_secs(10));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use temp_testdir::*;

    use tabby_common::config::{Config, Repository};
    use tabby_common::path::set_tabby_root;
    use tracing_test::traced_test;

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
