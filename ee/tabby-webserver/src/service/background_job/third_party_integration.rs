use std::sync::Arc;

use anyhow::Result;
use async_stream::stream;
use chrono::{DateTime, Utc};
use futures::{stream::BoxStream, StreamExt};
use issues::{list_github_issues, list_gitlab_issues};
use juniper::ID;
use pulls::{get_github_pull_doc, list_github_pull_states};
use serde::{Deserialize, Serialize};
use tabby_common::config::CodeRepository;
use tabby_index::public::{CodeIndexer, StructuredDoc, StructuredDocIndexer, StructuredDocState};
use tabby_inference::Embedding;
use tabby_schema::{
    integration::{Integration, IntegrationKind, IntegrationService},
    job::JobService,
    repository::{ProvidedRepository, ThirdPartyRepositoryService},
    CoreError,
};
use tracing::debug;

use super::{helper::Job, BackgroundJobEvent};

mod commits;
mod error;
mod issues;
mod pulls;

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
            .get_provided_repository(&self.repository_id)
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
        let code_repository = &CodeRepository::new(&authenticated_url, &repository.source_id());
        let mut code = CodeIndexer::default();
        code.refresh(embedding.clone(), code_repository).await?;

        logkit::info!(
            "Indexing recent commits for repository {}",
            repository.display_name
        );

        if let Err(err) = self
            .sync_commit_history(code_repository, embedding.clone())
            .await
        {
            integration_service
                .update_integration_sync_status(&integration.id, Some(err.to_string()))
                .await?;
            logkit::error!("Failed to sync commit history: {}", err);
            return Err(err);
        };

        logkit::info!(
            "Indexing documents for repository {}",
            repository.display_name
        );

        self.sync_issues(
            &integration,
            integration_service.clone(),
            &repository,
            embedding.clone(),
        )
        .await?;

        self.sync_pulls(&integration, integration_service, &repository, embedding)
            .await?;

        Ok(())
    }

    async fn sync_commit_history(
        &self,
        repository: &CodeRepository,
        embedding: Arc<dyn Embedding>,
    ) -> tabby_schema::Result<()> {
        commits::refresh(embedding.clone(), repository).await
    }

    async fn sync_pulls(
        &self,
        integration: &Integration,
        integration_service: Arc<dyn IntegrationService>,
        repository: &ProvidedRepository,
        embedding: Arc<dyn Embedding>,
    ) -> tabby_schema::Result<()> {
        let mut pull_state_stream = match fetch_all_pull_states(integration, repository).await {
            Ok(s) => s,
            Err(e) => {
                integration_service
                    .update_integration_sync_status(&integration.id, Some(e.to_string()))
                    .await?;
                logkit::error!("Failed to fetch pulls: {}", e);
                return Ok(());
            }
        };

        let mut count = 0;
        let mut num_updated = 0;

        let index = StructuredDocIndexer::new(embedding);
        while let Some((pull, state)) = pull_state_stream.next().await {
            count += 1;
            if count % 100 == 0 {
                logkit::info!(
                    "{} pull docs seen, {} pull docs updated",
                    count,
                    num_updated
                );
            }

            if !index.presync(&state).await {
                continue;
            }

            let pull_doc = fetch_pull_structured_doc(integration, repository, pull).await?;

            index.sync(pull_doc).await;
            num_updated += 1;
        }
        logkit::info!(
            "{} pull docs seen, {} pull docs updated",
            count,
            num_updated
        );
        index.commit();

        Ok(())
    }

    async fn sync_issues(
        &self,
        integration: &Integration,
        integration_service: Arc<dyn IntegrationService>,
        repository: &ProvidedRepository,
        embedding: Arc<dyn Embedding>,
    ) -> tabby_schema::Result<()> {
        let issue_stream = match fetch_all_issues(integration, repository).await {
            Ok(s) => s,
            Err(e) => {
                integration_service
                    .update_integration_sync_status(&integration.id, Some(e.to_string()))
                    .await?;
                logkit::error!("Failed to fetch issues: {}", e);
                return Err(e);
            }
        };

        let index = StructuredDocIndexer::new(embedding);
        stream! {
            let mut count = 0;
            let mut num_updated = 0;
            for await (state, doc) in issue_stream {
                if index.presync(&state).await && index.sync(doc).await {
                    num_updated += 1
                }
                count += 1;
                if count % 100 == 0 {
                    logkit::info!("{} issue docs seen, {} issue docs updated", count, num_updated);
                };
            }

            logkit::info!("{} issue docs seen, {} issue docs updated", count, num_updated);
            index.commit();
        }
        .count()
        .await;

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

async fn fetch_all_issues(
    integration: &Integration,
    repository: &ProvidedRepository,
) -> tabby_schema::Result<BoxStream<'static, (StructuredDocState, StructuredDoc)>> {
    let s: BoxStream<(StructuredDocState, StructuredDoc)> = match &integration.kind {
        IntegrationKind::Github | IntegrationKind::GithubSelfHosted => list_github_issues(
            &repository.source_id(),
            integration.api_base(),
            &repository.display_name,
            &integration.access_token,
        )
        .await?
        .boxed(),
        IntegrationKind::Gitlab | IntegrationKind::GitlabSelfHosted => list_gitlab_issues(
            &repository.source_id(),
            integration.api_base(),
            &repository.display_name,
            &integration.access_token,
        )
        .await?
        .boxed(),
    };

    Ok(s)
}

async fn fetch_all_pull_states(
    integration: &Integration,
    repository: &ProvidedRepository,
) -> tabby_schema::Result<BoxStream<'static, (pulls::Pull, StructuredDocState)>> {
    match &integration.kind {
        IntegrationKind::Github | IntegrationKind::GithubSelfHosted => Ok(list_github_pull_states(
            integration.api_base(),
            &repository.display_name,
            &integration.access_token,
        )
        .await?
        .boxed()),
        IntegrationKind::Gitlab | IntegrationKind::GitlabSelfHosted => Err(CoreError::Other(
            anyhow::anyhow!("Gitlab does not support pull requests yet"),
        )),
    }
}

async fn fetch_pull_structured_doc(
    integration: &Integration,
    repository: &ProvidedRepository,
    pull: pulls::Pull,
) -> Result<StructuredDoc> {
    match pull {
        pulls::Pull::GitHub(pull) => {
            get_github_pull_doc(
                &repository.source_id(),
                pull,
                integration.api_base(),
                &repository.display_name,
                &integration.access_token,
            )
            .await
        }
    }
}
