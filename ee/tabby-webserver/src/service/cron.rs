use std::time::Duration;

use anyhow::Result;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{error, warn};

use crate::service::db::DbConn;

async fn new_job_scheduler(jobs: Vec<Job>) -> Result<JobScheduler> {
    let scheduler = JobScheduler::new().await?;
    for job in jobs {
        scheduler.add(job).await?;
    }
    scheduler.start().await?;
    Ok(scheduler)
}

async fn new_refresh_token_job(db_conn: DbConn) -> Result<Job> {
    // job is run every 2 hours
    let job = Job::new_async("0 0 1/2 * * * *", move |_, _| {
        let db_conn = db_conn.clone();
        Box::pin(async move {
            let res = db_conn.delete_expired_token().await;
            if let Err(e) = res {
                error!("failed to delete expired token: {}", e);
            }
        })
    })?;

    Ok(job)
}

pub fn run_offline_job(db_conn: DbConn) {
    tokio::spawn(async move {
        let Ok(job) = new_refresh_token_job(db_conn.clone()).await else {
            error!("failed to create db job");
            return;
        };

        let Ok(mut scheduler) = new_job_scheduler(vec![job]).await else {
            error!("failed to start job scheduler");
            return;
        };

        loop {
            match scheduler.time_till_next_job().await {
                Ok(Some(duration)) => {
                    tokio::time::sleep(duration).await;
                }
                Ok(None) => {
                    warn!("no job available, exit scheduler");
                    return;
                }
                Err(e) => {
                    error!("failed to get job sleep time: {}, re-try in 1 second", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    });
}
