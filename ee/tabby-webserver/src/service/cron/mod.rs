mod db;

use std::time::Duration;

use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::error;

use crate::service::db::{DbConn, JobRun};

async fn new_job_scheduler(jobs: Vec<Job>) -> anyhow::Result<JobScheduler> {
    let scheduler = JobScheduler::new().await?;
    for job in jobs {
        scheduler.add(job).await?;
    }
    scheduler.start().await?;
    Ok(scheduler)
}

pub fn run_cron(db_conn: &DbConn) {
    let db_conn = db_conn.clone();
    tokio::spawn(async move {
        let Ok(job1) = db::refresh_token_job(db_conn.clone()).await else {
            error!("failed to create db job");
            return;
        };
        // run every 5 minutes
        let Ok(job2) = repository_job(db_conn.clone(), "sync".to_owned(), "0 1/5 * * * * *").await
        else {
            error!("failed to create sync job");
            return;
        };
        // run every 5 hours
        let Ok(job3) = repository_job(db_conn.clone(), "index".to_owned(), "0 0 1/5 * * * *").await
        else {
            error!("failed to create index job");
            return;
        };

        let Ok(mut scheduler) = new_job_scheduler(vec![job1, job2, job3]).await else {
            error!("failed to start job scheduler");
            return;
        };

        loop {
            match scheduler.time_till_next_job().await {
                Ok(Some(duration)) => {
                    tokio::time::sleep(duration).await;
                }
                Ok(None) => {
                    // wait until scheduler increases jobs' tick
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                Err(e) => {
                    error!("failed to get job sleep time: {}, re-try in 1 second", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    });
}

async fn repository_job(db_conn: DbConn, job_name: String, schedule: &str) -> anyhow::Result<Job> {
    let job = Job::new_async(schedule, move |_, _| {
        let job_name = job_name.clone();
        let db_conn = db_conn.clone();
        Box::pin(async move {
            // run command as a child process
            let start_time = chrono::Utc::now();
            let exe = std::env::current_exe().unwrap();
            let output = tokio::process::Command::new(exe)
                .arg(&format!("job::{}", &job_name))
                .output()
                .await;
            let Ok(output) = output else {
                error!("`{}` failed: {:?}", &job_name, output.unwrap_err());
                return;
            };
            let finish_time = chrono::Utc::now();

            // save run result to db
            let run = JobRun {
                id: 0,
                job_name,
                start_time,
                finish_time: Some(finish_time),
                exit_code: output.status.code(),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            };
            let res = db_conn.create_job_run(run).await;
            if let Err(e) = res {
                error!("failed to save job run result: {}", e);
            }
        })
    })?;

    Ok(job)
}
