use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use fetch::fetch_all_repos;
use juniper::ID;
use tabby_common::config::RepositoryConfig;
use tabby_db::DbConn;
use tabby_schema::{
    integration::{IntegrationKind, IntegrationService},
    repository::{ProvidedRepository, ThirdPartyRepositoryService},
    AsRowid, DbEnum, Result,
};
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, error};
use url::Url;

use crate::service::{background_job::BackgroundJobEvent, graphql_pagination_to_filter};

mod fetch;

struct ThirdPartyRepositoryServiceImpl {
    db: DbConn,
    integration: Arc<dyn IntegrationService>,
    background_job: UnboundedSender<BackgroundJobEvent>,
}

pub fn create(
    db: DbConn,
    integration: Arc<dyn IntegrationService>,
    background_job: UnboundedSender<BackgroundJobEvent>,
) -> impl ThirdPartyRepositoryService {
    ThirdPartyRepositoryServiceImpl {
        db,
        integration,
        background_job,
    }
}

#[async_trait]
impl ThirdPartyRepositoryService for ThirdPartyRepositoryServiceImpl {
    async fn list_repositories(
        &self,
        integration_ids: Option<Vec<ID>>,
        kind: Option<IntegrationKind>,
        active: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<ProvidedRepository>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;

        let integration_ids = integration_ids
            .into_iter()
            .flatten()
            .map(|id| id.as_rowid())
            .collect::<Result<Vec<_>, _>>()?;

        let kind = kind.map(|kind| kind.as_enum_str().to_string());

        Ok(self
            .db
            .list_provided_repositories(integration_ids, kind, active, limit, skip_id, backwards)
            .await?
            .into_iter()
            .map(ProvidedRepository::try_from)
            .collect::<Result<_, _>>()?)
    }

    async fn get_repository(&self, id: ID) -> Result<ProvidedRepository> {
        let repo = self.db.get_provided_repository(id.as_rowid()?).await?;
        Ok(repo.try_into()?)
    }

    async fn update_repository_active(&self, id: ID, active: bool) -> Result<()> {
        self.db
            .update_provided_repository_active(id.as_rowid()?, active)
            .await?;

        if active {
            let repo = self.get_repository(id).await?;
            let integration = self
                .integration
                .get_integration(repo.integration_id.clone())
                .await?;
            let git_url = format_authenticated_url(
                &integration.kind,
                &repo.git_url,
                &integration.access_token,
            )?;
            let _ = self
                .background_job
                .send(BackgroundJobEvent::Scheduler(RepositoryConfig::new(
                    git_url,
                )));
        }

        Ok(())
    }

    async fn list_active_git_urls(&self) -> Result<Vec<String>> {
        let mut urls = vec![];

        let integrations = self
            .integration
            .list_integrations(None, None, None, None, None, None)
            .await?;

        for integration in integrations {
            let repositories = self
                .list_repositories(
                    Some(vec![integration.id.clone()]),
                    None,
                    Some(true),
                    None,
                    None,
                    None,
                    None,
                )
                .await?;

            for repository in repositories {
                let url = format_authenticated_url(
                    &integration.kind,
                    &repository.git_url,
                    &integration.access_token,
                )?;
                urls.push(url);
            }
        }

        Ok(urls)
    }

    async fn sync_repositories(&self, integration_id: ID) -> Result<()> {
        refresh_repositories_for_provider(self, &*self.integration, integration_id).await?;
        Ok(())
    }

    async fn upsert_repository(
        &self,
        integration_id: ID,
        vendor_id: String,
        display_name: String,
        git_url: String,
    ) -> Result<()> {
        self.db
            .upsert_provided_repository(
                integration_id.as_rowid()?,
                vendor_id,
                display_name,
                git_url,
            )
            .await?;
        Ok(())
    }

    async fn delete_outdated_repositories(
        &self,
        integration_id: ID,
        before: DateTime<Utc>,
    ) -> Result<usize> {
        Ok(self
            .db
            .delete_outdated_provided_repositories(integration_id.as_rowid()?, before.into())
            .await?)
    }
}

async fn refresh_repositories_for_provider(
    repository: &dyn ThirdPartyRepositoryService,
    integration: &dyn IntegrationService,
    provider_id: ID,
) -> Result<()> {
    let provider = integration.get_integration(provider_id.clone()).await?;
    debug!(
        "Refreshing repositories for provider: {}",
        provider.display_name
    );

    let start = Utc::now();
    let repos = match fetch_all_repos(provider.kind.clone(), &provider.access_token).await {
        Ok(repos) => repos,
        Err((e, true)) => {
            integration
                .update_integration_error(provider.id.clone(), Some("".into()))
                .await?;
            error!(
                "Credentials for integration {} are expired or invalid",
                provider.display_name
            );
            return Err(e.into());
        }
        Err((e, false)) => {
            error!("Failed to fetch repositories from github: {e}");
            return Err(e.into());
        }
    };
    for repo in repos {
        debug!("importing: {}", repo.name);

        let id = repo.vendor_id;

        repository
            .upsert_repository(provider_id.clone(), id, repo.name, repo.git_url)
            .await?;
    }

    integration
        .update_integration_error(provider_id.clone(), None)
        .await?;
    let num_removed = repository
        .delete_outdated_repositories(provider_id, start)
        .await?;
    debug!("Removed {} outdated repositories", num_removed);
    Ok(())
}

fn format_authenticated_url(
    kind: &IntegrationKind,
    git_url: &str,
    access_token: &str,
) -> Result<String> {
    let mut url = Url::parse(git_url).map_err(anyhow::Error::from)?;
    match kind {
        IntegrationKind::Github => {
            let _ = url.set_username(access_token);
        }
        IntegrationKind::Gitlab => {
            let _ = url.set_username("oauth2");
            let _ = url.set_password(Some(access_token));
        }
    }
    Ok(url.to_string())
}
