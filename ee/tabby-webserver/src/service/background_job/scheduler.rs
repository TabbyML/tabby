use std::{process::Stdio, str::FromStr};

use anyhow::Context;
use apalis::{
    cron::{CronStream, Schedule},
    prelude::{Data, Job, Monitor, WorkerBuilder, WorkerFactoryFn},
    utils::TokioExecutor,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;
use tokio::io::AsyncBufReadExt;

use super::logger::{JobLogLayer, JobLogger};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SchedulerJob {}

impl Job for SchedulerJob {
    const NAME: &'static str = "scheduler";
}

impl SchedulerJob {
    async fn run_impl(
        self,
        job_logger: Data<JobLogger>,
        db: Data<DbConn>,
        local_port: Data<u16>,
    ) -> anyhow::Result<i32> {
        let local_port = *local_port;
        let exe = std::env::current_exe()?;

        let mut child = tokio::process::Command::new(exe)
            .arg("scheduler")
            .arg("--now")
            .arg("--url")
            .arg(format!("localhost:{local_port}"))
            .arg("--token")
            .arg(db.read_registration_token().await?)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        {
            // Pipe stdout
            let stdout = child.stdout.take().context("Failed to acquire stdout")?;
            let logger = job_logger.clone();
            tokio::spawn(async move {
                let stdout = tokio::io::BufReader::new(stdout);
                let mut stdout = stdout.lines();
                while let Ok(Some(line)) = stdout.next_line().await {
                    let _ = logger.stdout_writeline(line).await;
                }
            });
        }

        {
            // Pipe stderr
            let stderr = child.stderr.take().context("Failed to acquire stderr")?;
            let logger = job_logger.clone();
            tokio::spawn(async move {
                let stderr = tokio::io::BufReader::new(stderr);
                let mut stdout = stderr.lines();
                while let Ok(Some(line)) = stdout.next_line().await {
                    let _ = logger.stderr_writeline(line).await;
                }
            });
        }
        if let Some(exit_code) = child.wait().await.ok().and_then(|s| s.code()) {
            Ok(exit_code)
        } else {
            Ok(-1)
        }
    }

    async fn cron(
        _now: DateTime<Utc>,
        logger: Data<JobLogger>,
        db: Data<DbConn>,
        local_port: Data<u16>,
    ) -> crate::schema::Result<i32> {
        let job = SchedulerJob {};
        Ok(job.run_impl(logger, db, local_port).await?)
    }

    pub fn register(
        monitor: Monitor<TokioExecutor>,
        db: DbConn,
        local_port: u16,
    ) -> Monitor<TokioExecutor> {
        let schedule = Schedule::from_str("0 */10 * * * *").expect("unable to parse cron schedule");
        monitor.register(
            WorkerBuilder::new(SchedulerJob::NAME)
                .stream(CronStream::new(schedule).into_stream())
                .layer(JobLogLayer::new(db.clone(), SchedulerJob::NAME))
                .data(db)
                .data(local_port)
                .build_fn(SchedulerJob::cron),
        )
    }
}
