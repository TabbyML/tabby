use std::sync::Arc;

use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_common::config::RepositoryConfig;
use tabby_inference::Embedding;
use tabby_scheduler::CodeIndexer;
use tabby_schema::{job::JobService, repository::GitRepositoryService};

use super::{
    cprintln,
    helper::{Job, JobLogger},
    BackgroundJobEvent,
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SchedulerGitJob {
    repository: RepositoryConfig,
}

impl SchedulerGitJob {
    pub fn new(repository: RepositoryConfig) -> Self {
        Self { repository }
    }
}

impl Job for SchedulerGitJob {
    const NAME: &'static str = "scheduler_git";
}

impl SchedulerGitJob {
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
        git_repository: Arc<dyn GitRepositoryService>,
        job: Arc<dyn JobService>,
    ) -> tabby_schema::Result<()> {
        let repositories = git_repository
            .repository_list()
            .await
            .context("Must be able to retrieve repositories for sync")?;

        let repositories: Vec<_> = repositories
            .into_iter()
            .map(|repo| RepositoryConfig::new(repo.git_url))
            .collect();

        let mut code = CodeIndexer::default();

        code.garbage_collection(&repositories).await;

        for repository in repositories {
            let _ = job
                .trigger(BackgroundJobEvent::SchedulerGitRepository(repository).to_command())
                .await;
        }
        Ok(())
    }
}
