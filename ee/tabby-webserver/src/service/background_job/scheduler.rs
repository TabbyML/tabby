use std::sync::Arc;

use anyhow::Context;
use apalis::{
    prelude::{Data, Job, Monitor, Storage, WorkerFactoryFn},
    sqlite::{SqlitePool, SqliteStorage},
    utils::TokioExecutor,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
use tabby_db::DbConn;
use tabby_scheduler::CodeIndexer;

use super::{
    cprintln,
    helper::{BasicJob, CronJob, JobLogger},
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SchedulerJob {
    repository: RepositoryConfig,
}

impl SchedulerJob {
    pub fn new(repository: RepositoryConfig) -> Self {
        Self { repository }
    }
}

impl Job for SchedulerJob {
    const NAME: &'static str = "scheduler";
}

impl CronJob for SchedulerJob {
    const SCHEDULE: &'static str = "@hourly";
}

impl SchedulerJob {
    async fn run(self, job_logger: Data<JobLogger>) -> tabby_schema::Result<()> {
        let repository = self.repository.clone();
        tokio::spawn(async move {
            let mut code = CodeIndexer::default();
            cprintln!(
                job_logger,
                "Refreshing repository {}",
                repository.canonical_git_url()
            );
            code.refresh(&repository).await;
        })
        .await
        .context("Job execution failed")?;
        Ok(())
    }

    async fn cron(
        _now: DateTime<Utc>,
        repository_access: Data<Arc<dyn RepositoryAccess>>,
        storage: Data<SqliteStorage<SchedulerJob>>,
    ) -> tabby_schema::Result<()> {
        let repositories = repository_access
            .list_repositories()
            .await
            .context("Must be able to retrieve repositories for sync")?;

        let mut code = CodeIndexer::default();
        code.garbage_collection(&repositories);

        let mut storage = (*storage).clone();

        for repository in repositories {
            storage
                .push(SchedulerJob::new(repository))
                .await
                .context("unable to push job")?;
        }
        Ok(())
    }

    pub fn register(
        monitor: Monitor<TokioExecutor>,
        pool: SqlitePool,
        db: DbConn,
        repository_access: Arc<dyn RepositoryAccess>,
    ) -> (SqliteStorage<SchedulerJob>, Monitor<TokioExecutor>) {
        let storage = SqliteStorage::new(pool);
        let monitor = monitor
            .register(Self::basic_worker(storage.clone(), db.clone()).build_fn(Self::run))
            .register(
                Self::cron_worker(db.clone())
                    .data(storage.clone())
                    .data(repository_access)
                    .build_fn(SchedulerJob::cron),
            );
        (storage, monitor)
    }
}
