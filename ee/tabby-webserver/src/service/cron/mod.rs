mod db;
mod job_utils;

use tabby_db::DbConn;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::error;

async fn new_job_scheduler(jobs: Vec<Job>) -> anyhow::Result<JobScheduler> {
    let scheduler = JobScheduler::new().await?;
    for job in jobs {
        scheduler.add(job).await?;
    }
    scheduler.start().await?;
    Ok(scheduler)
}

pub async fn run_cron(db_conn: &DbConn, turn_on_scheduler_jobs: bool) {
    let db_conn = db_conn.clone();
    let mut jobs = vec![];

    let Ok(job1) = db::refresh_token_job(db_conn.clone()).await else {
        error!("failed to create db job");
        return;
    };
    jobs.push(job1);

    if turn_on_scheduler_jobs {
        // run every 5 minutes
        let Ok(job2) =
            job_utils::run_job(db_conn.clone(), "sync".to_owned(), "0 1/5 * * * * *").await
        else {
            error!("failed to create sync job");
            return;
        };
        jobs.push(job2);

        // run every 5 hours
        let Ok(job3) =
            job_utils::run_job(db_conn.clone(), "index".to_owned(), "0 0 1/5 * * * *").await
        else {
            error!("failed to create index job");
            return;
        };
        jobs.push(job3);
    }

    if new_job_scheduler(jobs).await.is_err() {
        error!("failed to start job scheduler");
    };
}
