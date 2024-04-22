use anyhow::Result;
use std::{collections::HashSet, sync::Arc};

use juniper::ID;

use crate::schema::github_repository_provider::GithubRepositoryProviderService;

async fn refresh_repositories(
    service: Arc<dyn GithubRepositoryProviderService>,
    provider_id: ID,
) -> Result<()> {
    let mut cached_repositories: HashSet<_> = self
        .list_github_provided_repositories_by_provider(vec![provider_id], None, None, None, None)
        .await?
        .into_iter()
        .map(|repo| repo.vendor_id)
        .collect();

    for provider in self
        .list_github_repository_providers(None, None, None, None)
        .await?
    {
        let repos = fetch_all_repos(&provider).await?;
        for repo in repos {
            let id = repo.id.to_string();
            let Some(mut url) = repo.git_url else {
                continue;
            };
            let _ = url.set_scheme("https");
            let url = url.to_string();
            // Remove IDs as we process them so the remaining IDs are ones that were not found in the listing
            if cached_repositories.remove(&id) {
                self.db
                    .update_github_provided_repository(id, repo.name, url)
                    .await?;
            } else {
                self.db
                    .create_github_provided_repository(
                        provider.id.clone().as_rowid()?,
                        id,
                        repo.name,
                        url,
                    )
                    .await?;
            }
        }
    }

    // Clean up repositories which were not returned by any listing
    for removed_repository in cached_repositories {
        self.db
            .delete_github_provided_repository_by_vendor_id(removed_repository)
            .await?;
    }

    Ok(())
}
