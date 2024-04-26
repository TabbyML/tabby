use std::{pin::Pin, process::Stdio, sync::Arc};

use anyhow::{Context, Result};
use futures::Future;
use tokio::io::AsyncBufReadExt;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{debug, error, info, warn};

use crate::schema::{job::JobService, worker::WorkerService};

pub async fn scheduler_job(
    job: Arc<dyn JobService>,
    worker: Arc<dyn WorkerService>,
    local_port: u16,
) -> anyhow::Result<Job> {
    let scheduler_mutex = Arc::new(tokio::sync::Mutex::new(()));

    let scheduler_job =
        move |uuid, mut scheduler: JobScheduler| -> Pin<Box<dyn Future<Output = ()> + Send>> {
            let worker = worker.clone();
            let job = job.clone();
            let scheduler_mutex = scheduler_mutex.clone();
            Box::pin(async move {
                let Ok(_guard) = scheduler_mutex.try_lock() else {
                    warn!("Scheduler job overlapped, skipping...");
                    return;
                };

                if let Err(err) = run_scheduler_now(job, worker, local_port).await {
                    error!("Failed to run scheduler job, reason: `{}`", err);
                }

                if let Ok(Some(next_tick)) = scheduler.next_tick_for_job(uuid).await {
                    debug!(
                        "Next time for scheduler job is {:?}",
                        next_tick.with_timezone(&chrono::Local)
                    );
                }
            })
        };

    let job = if std::env::var("TABBY_WEBSERVER_SCHEDULER_ONESHOT").is_ok() {
        warn!(
            "Running scheduler job as oneshot, this should only be used for debugging purpose..."
        );
        Job::new_one_shot_async(std::time::Duration::from_secs(10), scheduler_job)?
    } else {
        Job::new_async("0 1/10 * * * *", scheduler_job)?
    };

    Ok(job)
}

async fn run_scheduler_now(
    job: Arc<dyn JobService>,
    worker: Arc<dyn WorkerService>,
    local_port: u16,
) -> Result<()> {
    debug!("Running scheduler job...");
    let exe = std::env::current_exe()?;
    let job_id = job.start("scheduler".to_owned()).await?;

    let mut child = tokio::process::Command::new(exe)
        .arg("scheduler")
        .arg("--now")
        .arg("--url")
        .arg(format!("localhost:{local_port}"))
        .arg("--token")
        .arg(worker.read_registration_token().await?)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    {
        // Pipe stdout
        let job = job.clone();
        let job_id = job_id.clone();
        let stdout = child.stdout.take().context("Failed to acquire stdout")?;
        tokio::spawn(async move {
            let stdout = tokio::io::BufReader::new(stdout);
            let mut stdout = stdout.lines();
            while let Ok(Some(line)) = stdout.next_line().await {
                let _ = job.update_stdout(&job_id, line + "\n").await;
            }
        });
    }

    {
        // Pipe stderr
        let stderr = child.stderr.take().context("Failed to acquire stderr")?;
        let job = job.clone();
        let job_id = job_id.clone();
        tokio::spawn(async move {
            let stderr = tokio::io::BufReader::new(stderr);
            let mut stdout = stderr.lines();
            while let Ok(Some(line)) = stdout.next_line().await {
                let _ = job.update_stderr(&job_id, line + "\n").await;
            }
        });
    }
    if let Some(exit_code) = child.wait().await.ok().and_then(|s| s.code()) {
        job.complete(&job_id, exit_code).await?;
    } else {
        job.complete(&job_id, -1).await?;
    }

    Ok(())
}
