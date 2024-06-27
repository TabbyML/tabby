use std::sync::Arc;

use chrono::{DateTime, Utc};
use issues::{index_github_issues, index_gitlab_issues};
use juniper::ID;
use serde::{Deserialize, Serialize};
use tabby_common::config::RepositoryConfig;
use tabby_inference::Embedding;
use tabby_scheduler::{CodeIndexer, DocIndexer};
use tabby_schema::{
    integration::{IntegrationKind, IntegrationService},
    job::JobService,
    repository::ThirdPartyRepositoryService,
};
use tracing::debug;

use super::{helper::Job, BackgroundJobEvent};

mod issues;

#[derive(Serialize, Deserialize, Clone)]
pub struct SyncIntegrationJob {
    integration_id: ID,
}

impl Job for SyncIntegrationJob {
    const NAME: &'static str = "third_party_repository_sync";
}

impl SyncIntegrationJob {
    pub fn new(integration_id: ID) -> Self {
        Self { integration_id }
    }

    pub async fn run(
        self,
        repository_service: Arc<dyn ThirdPartyRepositoryService>,
    ) -> tabby_schema::Result<()> {
        repository_service
            .sync_repositories(self.integration_id)
            .await?;
        Ok(())
    }

    pub async fn cron(
        _now: DateTime<Utc>,
        integration: Arc<dyn IntegrationService>,
        job: Arc<dyn JobService>,
    ) -> tabby_schema::Result<()> {
        debug!("Syncing all github and gitlab repositories");

        for integration in integration
            .list_integrations(None, None, None, None, None, None)
            .await?
        {
            let _ = job
                .trigger(
                    BackgroundJobEvent::SyncThirdPartyRepositories(integration.id).to_command(),
                )
                .await;
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SchedulerGithubGitlabJob {
    repository_id: ID,
}

impl Job for SchedulerGithubGitlabJob {
    const NAME: &'static str = "scheduler_github_gitlab";
}

impl SchedulerGithubGitlabJob {
    pub fn new(repository_id: ID) -> Self {
        Self { repository_id }
    }

    pub async fn run(
        self,
        embedding: Arc<dyn Embedding>,
        repository_service: Arc<dyn ThirdPartyRepositoryService>,
        integration_service: Arc<dyn IntegrationService>,
    ) -> tabby_schema::Result<()> {
        let repository = repository_service
            .get_provided_repository(self.repository_id)
            .await?;
        let integration = integration_service
            .get_integration(repository.integration_id.clone())
            .await?;

        let authenticated_url = integration
            .kind
            .format_authenticated_url(&repository.git_url, &integration.access_token)?;

        // First, run the regular scheduler job to sync and index the repository
        logkit::info!(
            "Pulling source code for repository {}",
            repository.display_name
        );
        let mut code = CodeIndexer::default();
        code.refresh(embedding.clone(), &RepositoryConfig::new(authenticated_url))
            .await;

        logkit::info!("Indexing issues for repository {}", repository.display_name);

        let index = DocIndexer::new(embedding);
        match &integration.kind {
            IntegrationKind::Github | IntegrationKind::GithubSelfHosted => {
                index_github_issues(
                    &repository.source_id(),
                    integration.api_base(),
                    &repository.display_name,
                    &integration.access_token,
                    &index,
                )
                .await?;
            }
            IntegrationKind::Gitlab | IntegrationKind::GitlabSelfHosted => {
                index_gitlab_issues(
                    &repository.source_id(),
                    integration.api_base(),
                    &repository.display_name,
                    &integration.access_token,
                    &index,
                )
                .await?;
            }
        }
        index.commit();

        Ok(())
    }

    pub async fn cron(
        _now: DateTime<Utc>,
        repository: Arc<dyn ThirdPartyRepositoryService>,
        job: Arc<dyn JobService>,
    ) -> tabby_schema::Result<()> {
        for repository in repository
            .list_repositories_with_filter(None, None, Some(true), None, None, None, None)
            .await?
        {
            let _ = job
                .trigger(
                    BackgroundJobEvent::SchedulerGithubGitlabRepository(repository.id).to_command(),
                )
                .await;
        }
        Ok(())
    }
}
