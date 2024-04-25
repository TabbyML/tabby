use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use gitlab::{
    api::{projects::Projects, AsyncQuery, Pagination},
    GitlabBuilder,
};
use juniper::ID;
use serde::Deserialize;
use tracing::warn;

use crate::schema::gitlab_repository_provider::{
    GitlabRepositoryProvider, GitlabRepositoryProviderService,
};

pub async fn refresh_all_repositories(
    service: Arc<dyn GitlabRepositoryProviderService>,
) -> Result<()> {
    for provider in service
        .list_gitlab_repository_providers(vec![], None, None, None, None)
        .await?
    {
        let start = Utc::now();
        refresh_repositories_for_provider(service.clone(), provider.id.clone()).await?;
        service
            .delete_outdated_gitlab_provided_repositories(provider.id, start)
            .await?;
    }
    Ok(())
}

async fn refresh_repositories_for_provider(
    service: Arc<dyn GitlabRepositoryProviderService>,
    provider_id: ID,
) -> Result<()> {
    let provider = service.get_gitlab_repository_provider(provider_id).await?;
    let repos = match fetch_all_repos(&provider).await {
        Ok(repos) => repos,
        Err(e) if e.to_string().contains("401 Unauthorized") => {
            service
                .reset_gitlab_repository_provider_access_token(provider.id.clone())
                .await?;
            warn!(
                "GitLab credentials for provider {} are expired or invalid",
                provider.display_name
            );
            return Err(e);
        }
        Err(e) => {
            warn!("Failed to fetch repositories from github: {e}");
            return Err(e.into());
        }
    };
    for repo in repos {
        let id = repo.id.to_string();

        service
            .upsert_gitlab_provided_repository(
                provider.id.clone(),
                id,
                repo.name,
                repo.http_url_to_repo,
            )
            .await?;
    }

    Ok(())
}

#[derive(Deserialize)]
struct Repository {
    id: u128,
    name: String,
    http_url_to_repo: String,
}

async fn fetch_all_repos(
    provider: &GitlabRepositoryProvider,
) -> Result<Vec<Repository>, anyhow::Error> {
    let Some(token) = &provider.access_token else {
        return Ok(vec![]);
    };
    let gitlab = GitlabBuilder::new("gitlab.com", token)
        .build_async()
        .await?;

    Ok(gitlab::api::paged(
        Projects::builder().membership(true).build()?,
        Pagination::All,
    )
    .query_async(&gitlab)
    .await?)
}
