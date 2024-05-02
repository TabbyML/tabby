use std::{process::Stdio, str::FromStr};

use anyhow::Context;
use apalis::{
    cron::{CronStream, Schedule},
    prelude::{Data, Job, Monitor, Storage, WorkerBuilder, WorkerFactoryFn},
    sqlite::{SqlitePool, SqliteStorage},
    utils::TokioExecutor,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;
use tokio::io::AsyncBufReadExt;
use tower::limit::ConcurrencyLimitLayer;

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

    async fn run(
        self,
        logger: Data<JobLogger>,
        db: Data<DbConn>,
        local_port: Data<u16>,
    ) -> crate::schema::Result<i32> {
        Ok(self.run_impl(logger, db, local_port).await?)
    }

    async fn cron(
        _now: DateTime<Utc>,
        storage: Data<SqliteStorage<SchedulerJob>>,
    ) -> crate::schema::Result<()> {
        let mut storage = (*storage).clone();
        storage
            .push(SchedulerJob {})
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
        let schedule = Schedule::from_str("@hourly").expect("unable to parse cron schedule");
        let monitor = monitor
            .register(
                WorkerBuilder::new(Self::NAME)
                    .with_storage(storage.clone())
                    .layer(ConcurrencyLimitLayer::new(1))
                    .layer(JobLogLayer::new(db.clone(), Self::NAME))
                    .data(db.clone())
                    .data(local_port)
                    .build_fn(Self::run),
            )
            .register(
                WorkerBuilder::new(SchedulerJob::NAME)
                    .stream(CronStream::new(schedule).into_stream())
                    .data(storage.clone())
                    .build_fn(SchedulerJob::cron),
            );
        (storage, monitor)
    }
}
