use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::ID;
use tabby_db::DbConn;
use url::Url;

use super::{AsID, AsRowid};
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

pub fn create(db: DbConn) -> impl GithubRepositoryProviderService {
    GithubRepositoryProviderServiceImpl { db }
}

#[async_trait]
impl GithubRepositoryProviderService for GithubRepositoryProviderServiceImpl {
    async fn create_github_repository_provider(
        &self,
        display_name: String,
        application_id: String,
        application_secret: String,
    ) -> Result<ID> {
        let id = self
            .db
            .create_github_provider(display_name, application_id, application_secret)
            .await?;
        Ok(id.as_id())
    }

    async fn get_github_repository_provider(&self, id: ID) -> Result<GithubRepositoryProvider> {
        let provider = self.db.get_github_provider(id.as_rowid()?).await?;
        Ok(provider.into())
    }

    async fn delete_github_repository_provider(&self, id: ID) -> Result<()> {
        self.db.delete_github_provider(id.as_rowid()?).await?;
        Ok(())
    }

    async fn read_github_repository_provider_secret(&self, id: ID) -> Result<String> {
        let provider = self.db.get_github_provider(id.as_rowid()?).await?;
        Ok(provider.secret)
    }

    async fn update_github_repository_provider_access_token(
        &self,
        id: ID,
        access_token: Option<String>,
    ) -> Result<()> {
        self.db
            .update_github_provider_access_token(id.as_rowid()?, access_token)
            .await?;
        Ok(())
    }

    async fn list_github_repository_providers(
        &self,
        ids: Vec<ID>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GithubRepositoryProvider>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;

        let ids = ids
            .into_iter()
            .map(|id| id.as_rowid())
            .collect::<Result<Vec<_>, _>>()?;

        let providers = self
            .db
            .list_github_repository_providers(ids, limit, skip_id, backwards)
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

    async fn upsert_github_provided_repository(
        &self,
        provider_id: ID,
        vendor_id: String,
        display_name: String,
        git_url: String,
    ) -> Result<()> {
        self.db
            .upsert_github_provided_repository(
                provider_id.as_rowid()?,
                vendor_id,
                display_name,
                git_url,
            )
            .await?;
        Ok(())
    }

    async fn update_github_provided_repository_active(&self, id: ID, active: bool) -> Result<()> {
        self.db
            .update_github_provided_repository_active(id.as_rowid()?, active)
            .await?;
        Ok(())
    }

    async fn update_github_repository_provider(
        &self,
        id: ID,
        display_name: String,
        application_id: String,
        secret: Option<String>,
    ) -> Result<()> {
        self.db
            .update_github_provider(id.as_rowid()?, display_name, application_id, secret)
            .await?;
        Ok(())
    }

    async fn list_provided_git_urls(&self) -> Result<Vec<String>> {
        let tokens: HashMap<String, String> = self
            .list_github_repository_providers(vec![], None, None, None, None)
            .await?
            .into_iter()
            .filter_map(|provider| Some((provider.id.to_string(), provider.access_token?)))
            .collect();

        let mut repos = self
            .list_github_provided_repositories_by_provider(vec![], None, None, None, None)
            .await?;

        // Deduplicate by vendor ID
        let mut vendor_ids = HashSet::new();
        repos.retain(|repo| vendor_ids.insert(repo.vendor_id.clone()));

        let urls = repos
            .into_iter()
            .filter_map(|repo| {
                if !repo.active {
                    return None;
                }
                let mut url = Url::parse(&repo.git_url).ok()?;
                url.set_username(tokens.get(&repo.github_repository_provider_id.to_string())?)
                    .ok()?;
                Some(url.to_string())
            })
            .collect();

        Ok(urls)
    }

    async fn delete_outdated_github_provided_repositories(
        &self,
        provider_id: ID,
        cutoff_timestamp: DateTime<Utc>,
    ) -> Result<()> {
        self.db
            .delete_outdated_github_repositories(provider_id.as_rowid()?, cutoff_timestamp)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::AsID;

    #[tokio::test]
    async fn test_github_provided_repositories() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

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
            .upsert_github_provided_repository(
                provider_id1,
                "vendor_id1".into(),
                "test_repo1".into(),
                "https://github.com/test/test1".into(),
            )
            .await
            .unwrap();

        let repo_id2 = db
            .upsert_github_provided_repository(
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
    async fn test_github_repository_provider_crud() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = super::create(db.clone());

        let id = service
            .create_github_repository_provider("example".into(), "id".into(), "secret".into())
            .await
            .unwrap();

        // Test retrieving github provider by ID
        let provider1 = service
            .get_github_repository_provider(id.clone())
            .await
            .unwrap();
        assert_eq!(
            provider1,
            GithubRepositoryProvider {
                id: id.clone(),
                display_name: "example".into(),
                application_id: "id".into(),
                secret: "secret".into(),
                access_token: None,
                connected: false,
            }
        );

        // Test reading github provider secret
        let secret1 = service
            .read_github_repository_provider_secret(id.clone())
            .await
            .unwrap();
        assert_eq!(secret1, "secret");

        // Test listing github providers
        let providers = service
            .list_github_repository_providers(vec![], None, None, None, None)
            .await
            .unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].access_token, None);

        // Test updating github provider tokens
        service
            .update_github_repository_provider_access_token(id.clone(), Some("test_token".into()))
            .await
            .unwrap();

        assert_eq!(
            service
                .get_github_repository_provider(id.clone())
                .await
                .unwrap()
                .access_token,
            Some("test_token".into())
        );

        // Test updating github provider application ID / secret
        let id2 = service
            .create_github_repository_provider("example2".into(), "id2".into(), "secret".into())
            .await
            .unwrap();

        // Should fail: Duplicate application ID
        assert!(service
            .update_github_repository_provider(id2.clone(), "example2".into(), "id".into(), None)
            .await
            .is_err());

        service
            .update_github_repository_provider(
                id2.clone(),
                "example2".into(),
                "id2".into(),
                Some("secret2".into()),
            )
            .await
            .unwrap();

        assert_eq!(
            db.get_github_provider(id2.as_rowid().unwrap())
                .await
                .unwrap()
                .secret,
            "secret2"
        );

        // Test deleting github provider
        service
            .delete_github_repository_provider(id.clone())
            .await
            .unwrap();

        assert_eq!(
            1,
            service
                .list_github_repository_providers(vec![], None, None, None, None)
                .await
                .unwrap()
                .len()
        );
    }

    #[tokio::test]
    async fn test_provided_git_urls() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let provider_id = db
            .create_github_provider(
                "provider1".into(),
                "application_id1".into(),
                "secret1".into(),
            )
            .await
            .unwrap();

        let repo_id = db
            .upsert_github_provided_repository(
                provider_id,
                "vendor_id1".into(),
                "test_repo".into(),
                "https://github.com/TabbyML/tabby".into(),
            )
            .await
            .unwrap();

        db.update_github_provided_repository_active(repo_id, true)
            .await
            .unwrap();

        service
            .update_github_repository_provider_access_token(
                provider_id.as_id(),
                Some("token".into()),
            )
            .await
            .unwrap();

        let git_urls = service.list_provided_git_urls().await.unwrap();
        assert_eq!(
            git_urls,
            ["https://token@github.com/TabbyML/tabby".to_string()]
        );
    }
}
