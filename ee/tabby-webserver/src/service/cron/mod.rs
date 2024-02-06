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

pub async fn run_cron(db_conn: &DbConn) {
    let db_conn = db_conn.clone();
    let mut jobs = vec![];

    let Ok(job1) = db::refresh_token_job(db_conn.clone()).await else {
        error!("failed to create db job");
        return;
    };
    jobs.push(job1);

    if new_job_scheduler(jobs).await.is_err() {
        error!("failed to start job scheduler");
    };
}
