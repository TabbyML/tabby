use std::collections::HashMap;

use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use url::Url;

use super::AsRowid;
use crate::{
    schema::{
        github_repository_provider::{
            GithubProvidedRepository, GithubRepositoryProvider, GithubRepositoryProviderService,
        },
        Result,
    },
    service::graphql_pagination_to_filter,
};

struct GithubRepositoryProviderServiceImpl {
    db: DbConn,
}

pub fn new_github_repository_provider_service(db: DbConn) -> impl GithubRepositoryProviderService {
    GithubRepositoryProviderServiceImpl { db }
}

#[async_trait]
impl GithubRepositoryProviderService for GithubRepositoryProviderServiceImpl {
    async fn get_github_repository_provider(&self, id: ID) -> Result<GithubRepositoryProvider> {
        let provider = self.db.get_github_provider(id.as_rowid()?).await?;
        Ok(provider.into())
    }

    async fn update_github_repository_provider_access_token(
        &self,
        id: ID,
        access_token: String,
    ) -> Result<()> {
        self.db
            .update_github_provider_access_token(id.as_rowid()?, access_token)
            .await?;
        Ok(())
    }

    async fn list_github_repository_providers(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GithubRepositoryProvider>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let providers = self
            .db
            .list_github_repository_providers(limit, skip_id, backwards)
            .await?;
        Ok(providers
            .into_iter()
            .map(GithubRepositoryProvider::from)
            .collect())
    }

    async fn list_github_provided_repositories_by_provider(
        &self,
        providers: Vec<ID>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GithubProvidedRepository>> {
        let providers = providers
            .into_iter()
            .map(|i| i.as_rowid())
            .collect::<Result<Vec<_>, _>>()?;
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, last, first)?;
        let repos = self
            .db
            .list_github_provided_repositories(providers, limit, skip_id, backwards)
            .await?;

        Ok(repos
            .into_iter()
            .map(GithubProvidedRepository::from)
            .collect())
    }

    async fn update_github_provided_repository_active(&self, id: ID, active: bool) -> Result<()> {
        self.db
            .update_github_provided_repository_active(id.as_rowid()?, active)
            .await?;
        Ok(())
    }

    async fn list_provided_git_urls(&self) -> Result<Vec<String>> {
        let tokens: HashMap<String, String> = self
            .list_github_repository_providers(None, None, None, None)
            .await?
            .into_iter()
            .filter_map(|provider| Some((provider.id.to_string(), provider.access_token?)))
            .collect();

        let urls = self
            .list_github_provided_repositories_by_provider(vec![], None, None, None, None)
            .await?
            .into_iter()
            .filter_map(|repo| {
                let mut url = Url::parse(&repo.git_url).ok()?;
                url.set_username(tokens.get(&repo.github_repository_provider_id.to_string())?)
                    .ok()?;
                Some(url.to_string())
            })
            .collect();

        Ok(urls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::AsID;

    #[tokio::test]
    async fn test_github_provided_repositories() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = new_github_repository_provider_service(db.clone());

        let provider_id1 = db
            .create_github_provider(
                "test_provider1".into(),
                "test_id1".into(),
                "test_secret".into(),
            )
            .await
            .unwrap();

        let provider_id2 = db
            .create_github_provider(
                "test_provider2".into(),
                "test_id2".into(),
                "test_secret".into(),
            )
            .await
            .unwrap();

        let repo_id1 = db
            .create_github_provided_repository(
                provider_id1,
                "vendor_id1".into(),
                "test_repo1".into(),
                "https://github.com/test/test1".into(),
            )
            .await
            .unwrap();

        let repo_id2 = db
            .create_github_provided_repository(
                provider_id2,
                "vendor_id2".into(),
                "test_repo2".into(),
                "https://github.com/test/test2".into(),
            )
            .await
            .unwrap();

        // Test listing with no filter on providers
        let repos = service
            .list_github_provided_repositories_by_provider(vec![], None, None, None, None)
            .await
            .unwrap();

        assert_eq!(repos.len(), 2);
        assert_eq!(repos[0].name, "test_repo1");
        assert_eq!(repos[1].name, "test_repo2");

        // Test listing with a filter on providers
        let repos = service
            .list_github_provided_repositories_by_provider(
                vec![provider_id1.as_id()],
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(repos.len(), 1);
        assert_eq!(repos[0].name, "test_repo1");

        // Test deletion and toggling active status
        db.delete_github_provided_repository(repo_id1)
            .await
            .unwrap();

        db.update_github_provided_repository_active(repo_id2, true)
            .await
            .unwrap();

        let repos = service
            .list_github_provided_repositories_by_provider(vec![], None, None, None, None)
            .await
            .unwrap();

        assert_eq!(repos.len(), 1);
        assert!(repos[0].active);
    }

    #[tokio::test]
    async fn test_provided_git_urls() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = new_github_repository_provider_service(db.clone());

        let provider_id = db
            .create_github_provider(
                "provider1".into(),
                "application_id1".into(),
                "secret1".into(),
            )
            .await
            .unwrap();

        db.create_github_provided_repository(
            provider_id,
            "vendor_id1".into(),
            "test_repo".into(),
            "https://github.com/TabbyML/tabby".into(),
        )
        .await
        .unwrap();

        service
            .update_github_repository_provider_access_token(provider_id.as_id(), "token".into())
            .await
            .unwrap();

        let git_urls = service.list_provided_git_urls().await.unwrap();
        assert_eq!(
            git_urls,
            ["https://token@github.com/TabbyML/tabby".to_string()]
        );
    }
}
