use std::process::Stdio;

use anyhow::Context;
use apalis::{
    prelude::{Data, Job, Monitor, Storage, WorkerFactoryFn},
    sqlite::{SqlitePool, SqliteStorage},
    utils::TokioExecutor,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;
use tabby_schema::bail;
use tokio::io::AsyncBufReadExt;

use super::{
    ceprintln, cprintln,
    helper::{BasicJob, CronJob, JobLogger},
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SchedulerJob;

impl Job for SchedulerJob {
    const NAME: &'static str = "scheduler";
}

impl CronJob for SchedulerJob {
    const SCHEDULE: &'static str = "@hourly";
}

impl SchedulerJob {
    async fn run_impl(
        self,
        job_logger: Data<JobLogger>,
        db: Data<DbConn>,
        local_port: Data<u16>,
    ) -> anyhow::Result<()> {
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
                    cprintln!(logger, "{line}");
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
                    ceprintln!(logger, "{line}");
                }
            });
        }
        if let Some(exit_code) = child.wait().await.ok().and_then(|s| s.code()) {
            if exit_code != 0 {
                bail!("scheduler exited with code {exit_code}")
            }
        }

        Ok(())
    }

    async fn run(
        self,
        logger: Data<JobLogger>,
        db: Data<DbConn>,
        local_port: Data<u16>,
    ) -> tabby_schema::Result<()> {
        Ok(self.run_impl(logger, db, local_port).await?)
    }

    async fn cron(
        _now: DateTime<Utc>,
        storage: Data<SqliteStorage<SchedulerJob>>,
    ) -> tabby_schema::Result<()> {
        let mut storage = (*storage).clone();
        storage
            .push(SchedulerJob)
            .await
            .expect("unable to push job");
        Ok(())
    }

    pub fn register(
        monitor: Monitor<TokioExecutor>,
        pool: SqlitePool,
        db: DbConn,
        local_port: u16,
    ) -> (SqliteStorage<SchedulerJob>, Monitor<TokioExecutor>) {
        let storage = SqliteStorage::new(pool);
        let monitor = monitor
            .register(
                Self::basic_worker(storage.clone(), db.clone())
                    .data(local_port)
                    .build_fn(Self::run),
            )
            .register(
                Self::cron_worker(db.clone())
                    .data(storage.clone())
                    .build_fn(SchedulerJob::cron),
            );
        (storage, monitor)
    }
}
