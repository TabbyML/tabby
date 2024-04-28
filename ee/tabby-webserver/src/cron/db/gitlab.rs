use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use gitlab::{
    api::{projects::Projects, AsyncQuery, Pagination},
    GitlabBuilder,
};
use juniper::ID;
use serde::Deserialize;

use crate::{
    cron::controller::JobContext,
    schema::repository::{GitlabRepositoryProvider, GitlabRepositoryService},
};

pub async fn refresh_all_repositories(
    context: JobContext,
    service: Arc<dyn GitlabRepositoryService>,
) -> Result<i32> {
    for provider in service
        .list_providers(vec![], None, None, None, None)
        .await?
    {
        let start = Utc::now();
        context
            .stdout_writeline(format!(
                "Refreshing repositories for provider: {}\n",
                provider.display_name
            ))
            .await;
        refresh_repositories_for_provider(context.clone(), service.clone(), provider.id.clone())
            .await?;
        service
            .delete_outdated_repositories(provider.id, start)
            .await?;
    }
    Ok(0)
}

async fn refresh_repositories_for_provider(
    context: JobContext,
    service: Arc<dyn GitlabRepositoryService>,
    provider_id: ID,
) -> Result<()> {
    let provider = service.get_provider(provider_id).await?;
    let repos = match fetch_all_repos(&provider).await {
        Ok(repos) => repos,
        Err(e) if e.to_string().contains("401 Unauthorized") => {
            service
                .update_provider_status(provider.id.clone(), false)
                .await?;
            context
                .stderr_writeline(format!(
                    "GitLab credentials for provider {} are expired or invalid",
                    provider.display_name
                ))
                .await;
            return Err(e);
        }
        Err(e) => {
            context
                .stderr_writeline(format!("Failed to fetch repositories from gitlab: {e}"))
                .await;
            return Err(e);
        }
    };
    for repo in repos {
        context
            .stdout_writeline(format!("Importing: {}", &repo.name_with_namespace))
            .await;
        let id = repo.id.to_string();
        let url = repo.http_url_to_repo;
        let url = url.strip_suffix(".git").unwrap_or(&url);

        service
            .upsert_repository(
                provider.id.clone(),
                id,
                repo.name_with_namespace,
                url.to_string(),
            )
            .await?;
    }
    service
        .update_provider_status(provider.id.clone(), true)
        .await?;

    Ok(())
}

#[derive(Deserialize)]
struct Repository {
    id: u128,
    name_with_namespace: String,
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
