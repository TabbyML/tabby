use std::process::Stdio;

use tokio::{io::AsyncBufReadExt, process::Child};
use tokio_cron_scheduler::Job;
use tracing::error;

use crate::service::db::{DbConn, JobRun};

pub async fn run_job(db_conn: DbConn, job_name: String, schedule: &str) -> anyhow::Result<Job> {
    let job = Job::new_async(schedule, move |_, _| {
        let job_name = job_name.clone();
        let db_conn = db_conn.clone();
        Box::pin(async move {
            // create job run record
            let mut run = JobRun {
                job_name: job_name.clone(),
                start_time: chrono::Utc::now(),
                ..Default::default()
            };
            let res = db_conn.create_job_run(run.clone()).await;
            let Ok(job_id) = res else {
                error!(
                    "failed to create `{}` run record: {}",
                    job_name,
                    res.unwrap_err()
                );
                return;
            };
            run.id = job_id;

            // run command as a child process
            let exe = std::env::current_exe().unwrap();
            let mut child = tokio::process::Command::new(exe)
                .arg(&format!("job::{}", &job_name))
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();

            // spawn task to save stdout/stderr
            save_job_output(db_conn.clone(), job_id, &mut child);

            // wait process to exit
            run.exit_code = child.wait().await.ok().and_then(|s| s.code());
            run.finish_time = Some(chrono::Utc::now());
            db_conn
                .update_job_status(run.clone())
                .await
                .unwrap_or_else(|e| {
                    error!("failed to update `{}` run record: {}", job_name, e);
                });
        })
    })?;

    Ok(job)
}

/// Create two tasks to stream stdout/stderr from child process, into database
///
/// `child` must have piped stdout/stderr, or `unwrap` will panic on a `None` value
fn save_job_output(db_conn: DbConn, job_id: i32, child: &mut Child) {
    let stdout = child.stdout.take().unwrap();
    let stdout = tokio::io::BufReader::new(stdout);
    let mut stdout = stdout.lines();
    let db = db_conn.clone();
    tokio::spawn(async move {
        while let Ok(Some(mut line)) = stdout.next_line().await {
            line.push('\n');
            db.update_job_stdout(job_id, line).await.unwrap_or_default();
        }
    });

    let stderr = child.stderr.take().unwrap();
    let stderr = tokio::io::BufReader::new(stderr);
    let mut stderr = stderr.lines();
    let db = db_conn.clone();
    tokio::spawn(async move {
        while let Ok(Some(mut line)) = stderr.next_line().await {
            line.push('\n');
            db.update_job_stderr(job_id, line).await.unwrap_or_default();
        }
    });
}
