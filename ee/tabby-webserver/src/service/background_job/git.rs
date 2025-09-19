use std::sync::Arc;

use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_common::config::{config_index_to_id, CodeRepository, Config};
use tabby_index::public::CodeIndexer;
use tabby_inference::Embedding;
use tabby_schema::{job::JobService, repository::GitRepositoryService};
use tracing::debug;

use super::{
    calculate_current_shard, helper::Job, index_commits, should_process_repository,
    BackgroundJobEvent,
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SchedulerGitJob {
    repository: CodeRepository,
}

impl SchedulerGitJob {
    pub fn new(repository: CodeRepository) -> Self {
        Self { repository }
    }
}

impl Job for SchedulerGitJob {
    const NAME: &'static str = "scheduler_git";
}

impl SchedulerGitJob {
    pub async fn run(self, embedding: Arc<dyn Embedding>) -> tabby_schema::Result<()> {
        let repository = self.repository;
        tokio::spawn(async move {
            let mut code = CodeIndexer::default();
            if let Err(err) = code.refresh(embedding.clone(), &repository).await {
                logkit::warn!("Failed to refresh code index: {}", err);
                return Err(err.into());
            }

            if let Err(err) = index_commits::refresh(embedding, &repository).await {
                logkit::warn!("Failed to refresh commit index: {}", err);
                return Err(err);
            }

            Ok(())
        })
        .await
        .context("Job execution failed")??;

        Ok(())
    }

    pub async fn cron(
        now: DateTime<Utc>,
        git_repository: Arc<dyn GitRepositoryService>,
        job: Arc<dyn JobService>,
    ) -> tabby_schema::Result<()> {
        // Read git repositories from config file.
        let config_repositories = Config::load()
            .map(|config| config.repositories)
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, x)| CodeRepository::new(x.git_url(), &config_index_to_id(index)))
            .collect::<Vec<_>>();

        // Read git repositories from database.
        let repositories = git_repository
            .repository_list()
            .await
            .context("Must be able to retrieve repositories for sync")?;

        // Combine git repositories from config file and database.
        let repositories: Vec<_> = repositories
            .into_iter()
            .map(|repo| CodeRepository::new(&repo.git_url, &repo.source_id))
            .chain(config_repositories.into_iter())
            .collect();

        let number_of_repo = repositories.len();
        let current_shard = calculate_current_shard(number_of_repo, now.timestamp());

        for (i, repository) in repositories.iter().enumerate() {
            if !should_process_repository(i, current_shard, number_of_repo) {
                continue;
            }

            debug!("Scheduling git repository sync for {:?}", repository);
            let _ = job
                .trigger(
                    BackgroundJobEvent::SchedulerGitRepository(repository.clone()).to_command(),
                )
                .await;
        }
        Ok(())
    }
}
