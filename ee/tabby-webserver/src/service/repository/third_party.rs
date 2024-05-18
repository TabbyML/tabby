use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use fetch::fetch_all_repos;
use juniper::ID;
use strum::IntoEnumIterator;
use tabby_common::config::RepositoryConfig;
use tabby_db::DbConn;
use tabby_schema::{
    integration::{Integration, IntegrationKind, IntegrationService},
    repository::{ProvidedRepository, Repository, ThirdPartyRepositoryService},
    AsRowid, DbEnum, Result,
};
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, error};
use url::Url;

use self::fetch::RepositoryInfo;
use super::list_refs;
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

    async fn repository_list(&self) -> Result<Vec<Repository>> {
        let mut repos = vec![];
        for kind in IntegrationKind::iter() {
            let repos_for_kind = self
                .list_repositories(None, None, Some(true), None, None, None, None)
                .await?;

            repos.extend(repos_for_kind.into_iter().map(|repo| Repository {
                id: repo.id,
                name: repo.display_name,
                kind: kind.clone().into(),
                refs: list_refs(&repo.git_url),
                dir: RepositoryConfig::new(repo.git_url).dir(),
            }));
        }

        Ok(repos)
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
        let provider = self.integration.get_integration(integration_id).await?;
        debug!(
            "Refreshing repositories for provider: {}",
            provider.display_name
        );

        let repos = match fetch_all_repos(provider.kind.clone(), &provider.access_token).await {
            Ok(repos) => repos,
            Err((e, true)) => {
                self.integration
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

        refresh_repositories_for_provider(self, &*self.integration, provider, repos).await?;
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
    provider: Integration,
    repos: Vec<RepositoryInfo>,
) -> Result<()> {
    let start = Utc::now();

    for repo in repos {
        debug!("importing: {}", repo.name);

        let id = repo.vendor_id;

        repository
            .upsert_repository(provider.id.clone(), id, repo.name, repo.git_url)
            .await?;
    }

    integration
        .update_integration_error(provider.id.clone(), None)
        .await?;
    let num_removed = repository
        .delete_outdated_repositories(provider.id, start)
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

#[cfg(test)]
mod tests {

    use std::time::Duration;

    use super::*;

    async fn create_fake() -> (
        Arc<dyn ThirdPartyRepositoryService>,
        Arc<dyn IntegrationService>,
    ) {
        let (sender, _) = tokio::sync::mpsc::unbounded_channel();
        let db = DbConn::new_in_memory().await.unwrap();
        let integration = Arc::new(crate::integration::create(db.clone(), sender.clone()));
        let repository = Arc::new(create(db.clone(), integration.clone(), sender.clone()));
        (repository, integration)
    }

    #[tokio::test]
    async fn test_integrated_repositories() {
        let (repository, integration) = create_fake().await;

        let provider_id1 = integration
            .create_integration(
                IntegrationKind::Github,
                "test_id1".into(),
                "test_secret".into(),
            )
            .await
            .unwrap();

        let provider_id2 = integration
            .create_integration(
                IntegrationKind::Github,
                "test_id2".into(),
                "test_secret".into(),
            )
            .await
            .unwrap();

        repository
            .upsert_repository(
                provider_id1.clone(),
                "vendor_id1".into(),
                "test_repo1".into(),
                "https://github.com/test/test1".into(),
            )
            .await
            .unwrap();

        repository
            .upsert_repository(
                provider_id2,
                "vendor_id2".into(),
                "test_repo2".into(),
                "https://github.com/test/test2".into(),
            )
            .await
            .unwrap();

        // Test listing with no filter on providers
        let repos = repository
            .list_repositories(None, None, None, None, None, None, None)
            .await
            .unwrap();

        assert_eq!(repos.len(), 2);
        assert_eq!(repos[0].display_name, "test_repo1");
        assert_eq!(repos[1].display_name, "test_repo2");

        // Test listing with a filter on providers
        let repos = repository
            .list_repositories(
                Some(vec![provider_id1]),
                Some(IntegrationKind::Github),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(repos.len(), 1);
        assert_eq!(repos[0].display_name, "test_repo1");

        // Test listing with a filter on active status
        let repos = repository
            .list_repositories(None, None, Some(true), None, None, None, None)
            .await
            .unwrap();

        assert_eq!(0, repos.len());

        let repos = repository
            .list_repositories(None, None, Some(false), None, None, None, None)
            .await
            .unwrap();

        assert_eq!(2, repos.len());

        let repo_id = repos[0].id.clone();

        // Test toggling active status
        repository
            .update_repository_active(repo_id, true)
            .await
            .unwrap();

        let repos = repository
            .list_repositories(None, None, Some(true), None, None, None, None)
            .await
            .unwrap();

        assert_eq!(repos.len(), 1);
        assert!(repos[0].active);
    }

    #[tokio::test]
    async fn test_provided_git_urls() {
        let (repository, integration) = create_fake().await;

        let provider_id = integration
            .create_integration(IntegrationKind::Github, "provider1".into(), "token".into())
            .await
            .unwrap();

        repository
            .upsert_repository(
                provider_id,
                "vendor_id1".into(),
                "test_repo".into(),
                "https://github.com/TabbyML/tabby".into(),
            )
            .await
            .unwrap();

        let repo_id = repository
            .list_repositories(None, None, None, None, None, None, None)
            .await
            .unwrap()[0]
            .id
            .clone();

        repository
            .update_repository_active(repo_id.clone(), true)
            .await
            .unwrap();

        // Test github urls are formatted correctly
        let git_urls = repository.list_active_git_urls().await.unwrap();
        assert_eq!(
            git_urls,
            ["https://token@github.com/TabbyML/tabby".to_string()]
        );

        repository
            .update_repository_active(repo_id, false)
            .await
            .unwrap();

        // Test gitlab urls are formatted properly
        let provider_id2 = integration
            .create_integration(IntegrationKind::Gitlab, "provider2".into(), "token2".into())
            .await
            .unwrap();

        repository
            .upsert_repository(
                provider_id2,
                "vendor_id2".into(),
                "test_repo".into(),
                "https://gitlab.com/TabbyML/tabby".into(),
            )
            .await
            .unwrap();

        let repo_id = repository
            .list_repositories(
                None,
                Some(IntegrationKind::Gitlab),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap()[0]
            .id
            .clone();

        repository
            .update_repository_active(repo_id, true)
            .await
            .unwrap();

        let git_urls = repository.list_active_git_urls().await.unwrap();
        assert_eq!(
            git_urls,
            ["https://oauth2:token2@gitlab.com/TabbyML/tabby".to_string()]
        );
    }

    #[tokio::test]
    async fn test_refresh_repositories_for_provider() {
        let (repository, integration) = create_fake().await;

        let provider_id = integration
            .create_integration(IntegrationKind::Github, "gh".into(), "token".into())
            .await
            .unwrap();

        repository
            .upsert_repository(
                provider_id.clone(),
                "vendor_id".into(),
                "abc/def".into(),
                "https://github.com/abc/def".into(),
            )
            .await
            .unwrap();

        repository
            .upsert_repository(
                provider_id.clone(),
                "vendor_id2".into(),
                "repo2".into(),
                "https://github.com/TabbyML/tabby".into(),
            )
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_secs(1)).await;

        let mut new_repos = vec![];
        new_repos.push(RepositoryInfo {
            name: "TabbyML/tabby2".into(),
            git_url: "https://github.com/TabbyML/tabby".into(),
            vendor_id: "vendor_id2".into(),
        });

        new_repos.push(RepositoryInfo {
            name: "TabbyML/newrepo".into(),
            git_url: "https://github.com/TabbyML/newrepo".into(),
            vendor_id: "vendor_id3".into(),
        });

        let provider = integration.get_integration(provider_id).await.unwrap();
        refresh_repositories_for_provider(&*repository, &*integration, provider, new_repos)
            .await
            .unwrap();

        let repos = repository
            .list_repositories(None, None, None, None, None, None, None)
            .await
            .unwrap();

        assert_eq!(
            repos.iter().map(|r| &r.display_name).collect::<Vec<_>>(),
            vec!["TabbyML/tabby2", "TabbyML/newrepo"]
        );
    }
}
