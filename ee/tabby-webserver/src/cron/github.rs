use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use juniper::ID;
use octocrab::{models::Repository, GitHubError, Octocrab};
use tracing::warn;

use crate::schema::github_repository_provider::{
    GithubRepositoryProvider, GithubRepositoryProviderService,
};

pub async fn refresh_all_repositories(
    service: Arc<dyn GithubRepositoryProviderService>,
) -> Result<()> {
    for provider in service
        .list_github_repository_providers(vec![], None, None, None, None)
        .await?
    {
        let start = Utc::now();
        refresh_repositories_for_provider(service.clone(), provider.id.clone()).await?;
        service
            .delete_outdated_github_provided_repositories(provider.id, start)
            .await?;
    }
    Ok(())
}

async fn refresh_repositories_for_provider(
    service: Arc<dyn GithubRepositoryProviderService>,
    provider_id: ID,
) -> Result<()> {
    let provider = service.get_github_repository_provider(provider_id).await?;
    let repos = match fetch_all_repos(&provider).await {
        Ok(repos) => repos,
        Err(octocrab::Error::GitHub {
            source: source @ GitHubError { .. },
            ..
        }) if source.status_code.is_client_error() => {
            service
                .reset_github_repository_provider_access_token(provider.id.clone())
                .await?;
            warn!(
                "GitHub credentials for provider {} are expired or invalid",
                provider.display_name
            );
            return Err(source.into());
        }
        Err(e) => {
            warn!("Failed to fetch repositories from github: {e}");
            return Err(e.into());
        }
    };
    for repo in repos {
        let id = repo.id.to_string();
        let Some(url) = repo.git_url else {
            continue;
        };
        let url = url.to_string();
        let url = url
            .strip_prefix("git://")
            .map(|url| format!("https://{url}"))
            .unwrap_or(url);
        let url = url.strip_suffix(".git").unwrap_or(&url);

        service
            .upsert_github_provided_repository(
                provider.id.clone(),
                id,
                repo.full_name.unwrap_or(repo.name),
                url.to_string(),
            )
            .await?;
    }

    Ok(())
}

// FIXME(wsxiaoys): Convert to async stream
async fn fetch_all_repos(
    provider: &GithubRepositoryProvider,
) -> Result<Vec<Repository>, octocrab::Error> {
    let Some(token) = &provider.access_token else {
        return Ok(vec![]);
    };
    let octocrab = Octocrab::builder()
        .user_access_token(token.to_string())
        .build()?;

    let mut page = 1;
    let mut repos = vec![];

    loop {
        let response = octocrab
            .current()
            .list_repos_for_authenticated_user()
            .visibility("all")
            .page(page)
            .send()
            .await?;

        let pages = response.number_of_pages().unwrap_or_default() as u8;
        repos.extend(response.items);

        page += 1;
        if page > pages {
            break;
        }
    }
    Ok(repos)
}
