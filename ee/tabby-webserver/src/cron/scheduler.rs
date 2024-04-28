use std::{process::Stdio, sync::Arc};

use anyhow::{Context, Result};
use tokio::io::AsyncBufReadExt;
use tracing::debug;

use super::controller::{JobContext, JobController};
use crate::schema::worker::WorkerService;

pub async fn register(
    controller: &mut JobController,
    worker: Arc<dyn WorkerService>,
    local_port: u16,
) {
    controller
        .register_public("scheduler", "0 */10 * * * *", move |context| {
            let context = context.clone();
            let worker = worker.clone();
            Box::pin(async move { run_scheduler_now(context, worker, local_port).await })
        })
        .await;
}

async fn run_scheduler_now(
    context: JobContext,
    worker: Arc<dyn WorkerService>,
    local_port: u16,
) -> Result<i32> {
    debug!("Running scheduler job...");
    let exe = std::env::current_exe()?;

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
        let stdout = child.stdout.take().context("Failed to acquire stdout")?;
        let ctx = context.clone();
        tokio::spawn(async move {
            let stdout = tokio::io::BufReader::new(stdout);
            let mut stdout = stdout.lines();
            while let Ok(Some(line)) = stdout.next_line().await {
                let _ = ctx.stdout_writeline(line).await;
            }
        });
    }

    {
        // Pipe stderr
        let stderr = child.stderr.take().context("Failed to acquire stderr")?;
        let ctx = context.clone();
        tokio::spawn(async move {
            let stderr = tokio::io::BufReader::new(stderr);
            let mut stdout = stderr.lines();
            while let Ok(Some(line)) = stdout.next_line().await {
                let _ = ctx.stderr_writeline(line).await;
            }
        });
    }
    if let Some(exit_code) = child.wait().await.ok().and_then(|s| s.code()) {
        Ok(exit_code)
    } else {
        Ok(-1)
    }
}
