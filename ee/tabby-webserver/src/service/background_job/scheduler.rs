use std::sync::Arc;

use anyhow::Context;
use apalis::{
    prelude::{Data, Job, Monitor, Storage, WorkerFactoryFn},
    sqlite::{SqlitePool, SqliteStorage},
    utils::TokioExecutor,
};
use apalis_sql::Config;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_common::config::{ConfigAccess, RepositoryConfig};
use tabby_db::DbConn;
use tabby_inference::Embedding;
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
    async fn run(
        self,
        job_logger: Data<JobLogger>,
        embedding: Data<Arc<dyn Embedding>>,
    ) -> tabby_schema::Result<()> {
        let repository = self.repository.clone();
        tokio::spawn(async move {
            let mut code = CodeIndexer::default();
            cprintln!(
                job_logger,
                "Refreshing repository {}",
                repository.canonical_git_url()
            );
            code.refresh((*embedding).clone(), &repository).await;
        })
        .await
        .context("Job execution failed")?;
        Ok(())
    }

    async fn cron(
        _now: DateTime<Utc>,
        config_access: Data<Arc<dyn ConfigAccess>>,
        storage: Data<SqliteStorage<SchedulerJob>>,
    ) -> tabby_schema::Result<()> {
        let repositories = config_access
            .repositories()
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
        config: Config,
        config_access: Arc<dyn ConfigAccess>,
        embedding: Arc<dyn Embedding>,
    ) -> (SqliteStorage<SchedulerJob>, Monitor<TokioExecutor>) {
        let storage = SqliteStorage::new_with_config(pool, config);
        let monitor = monitor
            .register(
                Self::basic_worker(storage.clone(), db.clone())
                    .data(embedding)
                    .build_fn(Self::run),
            )
            .register(
                Self::cron_worker(db.clone())
                    .data(storage.clone())
                    .data(config_access)
                    .build_fn(SchedulerJob::cron),
            );
        (storage, monitor)
    }
}
