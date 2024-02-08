use std::{process::Stdio, sync::Arc, time::Duration};

use tokio::io::AsyncBufReadExt;
use tokio_cron_scheduler::Job;
use tracing::{info, warn};

use crate::schema::{job::JobService, worker::WorkerService};

pub async fn scheduler_job(
    job: Arc<dyn JobService>,
    worker: Arc<dyn WorkerService>,
    local_port: u16,
) -> anyhow::Result<Job> {
    let scheduler_mutex = Arc::new(tokio::sync::Mutex::new(()));

    // let job = Job::new_async("0 1/10 * * * *", move |uuid, mut scheduler| {
    let job = Job::new_one_shot_async(Duration::from_secs(1), move |uuid, mut scheduler| {
        let worker = worker.clone();
        let job = job.clone();
        let scheduler_mutex = scheduler_mutex.clone();
        Box::pin(async move {
            let Ok(_guard) = scheduler_mutex.try_lock() else {
                warn!("Scheduler job overlapped, skipping...");
                return;
            };

            info!("Running scheduler job...");
            let exe = std::env::current_exe().unwrap();
            let job_id = job.create_job_run("scheduler".to_owned()).await.unwrap();

            let mut child = tokio::process::Command::new(exe)
                .arg("scheduler")
                .arg("--now")
                .arg("--url")
                .arg(format!("localhost:{local_port}"))
                .arg("--token")
                .arg(worker.read_registration_token().await.unwrap())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();

            {
                // Pipe stdout
                let job = job.clone();
                let job_id = job_id.clone();
                let stdout = child.stdout.take().unwrap();
                tokio::spawn(async move {
                    let stdout = tokio::io::BufReader::new(stdout);
                    let mut stdout = stdout.lines();
                    while let Ok(Some(line)) = stdout.next_line().await {
                        println!("{line}");
                        let _ = job.update_job_stdout(&job_id, line + "\n").await;
                    }
                });
            }

            {
                // Pipe stderr
                let stderr = child.stderr.take().unwrap();
                let job = job.clone();
                let job_id = job_id.clone();
                tokio::spawn(async move {
                    let stderr = tokio::io::BufReader::new(stderr);
                    let mut stdout = stderr.lines();
                    while let Ok(Some(line)) = stdout.next_line().await {
                        eprintln!("{line}");
                        let _ = job.update_job_stderr(&job_id, line + "\n").await;
                    }
                });
            }
            if let Some(exit_code) = child.wait().await.ok().and_then(|s| s.code()) {
                let _ = job.complete_job_run(&job_id, exit_code).await;
            } else {
                let _ = job.complete_job_run(&job_id, -1).await;
            }

            if let Ok(Some(next_tick)) = scheduler.next_tick_for_job(uuid).await {
                info!(
                    "Next time for scheduler job is {:?}",
                    next_tick.with_timezone(&chrono::Local)
                );
            }
        })
    })?;

    Ok(job)
}
