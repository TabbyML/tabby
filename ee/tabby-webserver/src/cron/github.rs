use anyhow::Result;
use chrono::Utc;
use std::sync::Arc;

use crate::schema::github_repository_provider::GithubRepositoryProviderService;

pub async fn refresh_all_repositories(
    service: Arc<dyn GithubRepositoryProviderService>,
) -> Result<()> {
    for provider in service
        .list_github_repository_providers(None, None, None, None)
        .await?
    {
        let start = Utc::now();
        service.refresh_repositories(provider.id.clone()).await?;
        service
            .delete_outdated_github_provided_repositories(provider.id, start)
            .await?;
    }
    Ok(())
}
