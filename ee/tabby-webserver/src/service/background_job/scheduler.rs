use std::sync::Arc;

use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_common::config::{ConfigAccess, RepositoryConfig};
use tabby_inference::Embedding;
use tabby_scheduler::CodeIndexer;

use super::{
    cprintln,
    helper::{Job, JobLogger},
    BackgroundJobEvent,
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

impl SchedulerJob {
    pub async fn run(
        self,
        job_logger: JobLogger,
        embedding: Arc<dyn Embedding>,
    ) -> tabby_schema::Result<()> {
        let repository = self.repository.clone();
        tokio::spawn(async move {
            let mut code = CodeIndexer::default();
            cprintln!(
                job_logger,
                "Refreshing repository {}",
                repository.canonical_git_url()
            );
            code.refresh(embedding, &repository).await;
        })
        .await
        .context("Job execution failed")?;
        Ok(())
    }

    pub async fn cron(
        _now: DateTime<Utc>,
        config_access: Arc<dyn ConfigAccess>,
        sender: tokio::sync::mpsc::UnboundedSender<BackgroundJobEvent>,
    ) -> tabby_schema::Result<()> {
        let repositories = config_access
            .repositories()
            .await
            .context("Must be able to retrieve repositories for sync")?;

        let mut code = CodeIndexer::default();

        code.garbage_collection(&repositories);

        for repository in repositories {
            sender
                .send(BackgroundJobEvent::Scheduler(repository))
                .context("Failed to enqueue scheduler job")?;
        }
        Ok(())
    }
}
