use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::ID;
use tabby_db::DbConn;
use url::Url;

use super::{AsID, AsRowid};
use crate::{
    schema::{
        gitlab_repository_provider::{
            GitlabProvidedRepository, GitlabRepositoryProvider, GitlabRepositoryProviderService,
        },
        Result,
    },
    service::graphql_pagination_to_filter,
};

struct GitlabRepositoryProviderServiceImpl {
    db: DbConn,
}

pub fn create(db: DbConn) -> impl GitlabRepositoryProviderService {
    GitlabRepositoryProviderServiceImpl { db }
}

#[async_trait]
impl GitlabRepositoryProviderService for GitlabRepositoryProviderServiceImpl {
    async fn create_gitlab_repository_provider(
        &self,
        display_name: String,
        access_token: String,
    ) -> Result<ID> {
        let id = self
            .db
            .create_gitlab_provider(display_name, access_token)
            .await?;
        Ok(id.as_id())
    }

    async fn get_gitlab_repository_provider(&self, id: ID) -> Result<GitlabRepositoryProvider> {
        let provider = self.db.get_gitlab_provider(id.as_rowid()?).await?;
        Ok(provider.into())
    }

    async fn delete_gitlab_repository_provider(&self, id: ID) -> Result<()> {
        self.db.delete_gitlab_provider(id.as_rowid()?).await?;
        Ok(())
    }

    async fn reset_gitlab_repository_provider_access_token(&self, id: ID) -> Result<()> {
        self.db
            .reset_gitlab_provider_access_token(id.as_rowid()?)
            .await?;
        Ok(())
    }

    async fn list_gitlab_repository_providers(
        &self,
        ids: Vec<ID>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GitlabRepositoryProvider>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;

        let ids = ids
            .into_iter()
            .map(|id| id.as_rowid())
            .collect::<Result<Vec<_>, _>>()?;

        let providers = self
            .db
            .list_gitlab_repository_providers(ids, limit, skip_id, backwards)
            .await?;
        Ok(providers
            .into_iter()
            .map(GitlabRepositoryProvider::from)
            .collect())
    }

    async fn list_gitlab_provided_repositories_by_provider(
        &self,
        providers: Vec<ID>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GitlabProvidedRepository>> {
        let providers = providers
            .into_iter()
            .map(|i| i.as_rowid())
            .collect::<Result<Vec<_>, _>>()?;
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let repos = self
            .db
            .list_gitlab_provided_repositories(providers, limit, skip_id, backwards)
            .await?;

        Ok(repos
            .into_iter()
            .map(GitlabProvidedRepository::from)
            .collect())
    }

    async fn upsert_gitlab_provided_repository(
        &self,
        provider_id: ID,
        vendor_id: String,
        display_name: String,
        git_url: String,
    ) -> Result<()> {
        self.db
            .upsert_gitlab_provided_repository(
                provider_id.as_rowid()?,
                vendor_id,
                display_name,
                git_url,
            )
            .await?;
        Ok(())
    }

    async fn update_gitlab_provided_repository_active(&self, id: ID, active: bool) -> Result<()> {
        self.db
            .update_gitlab_provided_repository_active(id.as_rowid()?, active)
            .await?;
        Ok(())
    }

    async fn update_gitlab_repository_provider(
        &self,
        id: ID,
        display_name: String,
        access_token: String,
    ) -> Result<()> {
        self.db
            .update_gitlab_provider(id.as_rowid()?, display_name, access_token)
            .await?;
        Ok(())
    }

    async fn list_provided_git_urls(&self) -> Result<Vec<String>> {
        let tokens: HashMap<String, String> = self
            .list_gitlab_repository_providers(vec![], None, None, None, None)
            .await?
            .into_iter()
            .filter_map(|provider| Some((provider.id.to_string(), provider.access_token?)))
            .collect();

        let mut repos = self
            .list_gitlab_provided_repositories_by_provider(vec![], None, None, None, None)
            .await?;

        deduplicate_gitlab_repositories(&mut repos);

        let urls = repos
            .into_iter()
            .filter_map(|repo| {
                if !repo.active {
                    return None;
                }
                let mut url = Url::parse(&repo.git_url).ok()?;
                url.set_username("oauth2").ok()?;
                url.set_password(Some(
                    tokens.get(&repo.gitlab_repository_provider_id.to_string())?,
                ))
                .ok()?;
                Some(url.to_string())
            })
            .collect();

        Ok(urls)
    }

    async fn delete_outdated_gitlab_provided_repositories(
        &self,
        provider_id: ID,
        cutoff_timestamp: DateTime<Utc>,
    ) -> Result<()> {
        self.db
            .delete_outdated_gitlab_repositories(provider_id.as_rowid()?, cutoff_timestamp.into())
            .await?;
        Ok(())
    }
}

fn deduplicate_gitlab_repositories(repositories: &mut Vec<GitlabProvidedRepository>) {
    let mut vendor_ids = HashSet::new();
    repositories.retain(|repo| vendor_ids.insert(repo.vendor_id.clone()));
}

#[cfg(test)]
mod tests {
    use chrono::Duration;

    use super::*;
    use crate::service::AsID;

    #[tokio::test]
    async fn test_gitlab_provided_repositories() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let provider_id1 = db
            .create_gitlab_provider("test_id1".into(), "test_secret".into())
            .await
            .unwrap();

        let provider_id2 = db
            .create_gitlab_provider("test_id2".into(), "test_secret".into())
            .await
            .unwrap();

        let repo_id1 = db
            .upsert_gitlab_provided_repository(
                provider_id1,
                "vendor_id1".into(),
                "test_repo1".into(),
                "https://gitlab.com/test/test1".into(),
            )
            .await
            .unwrap();

        let repo_id2 = db
            .upsert_gitlab_provided_repository(
                provider_id2,
                "vendor_id2".into(),
                "test_repo2".into(),
                "https://gitlab.com/test/test2".into(),
            )
            .await
            .unwrap();

        // Test listing with no filter on providers
        let repos = service
            .list_gitlab_provided_repositories_by_provider(vec![], None, None, None, None)
            .await
            .unwrap();

        assert_eq!(repos.len(), 2);
        assert_eq!(repos[0].name, "test_repo1");
        assert_eq!(repos[1].name, "test_repo2");

        // Test listing with a filter on providers
        let repos = service
            .list_gitlab_provided_repositories_by_provider(
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
        db.delete_gitlab_provided_repository(repo_id1)
            .await
            .unwrap();

        db.update_gitlab_provided_repository_active(repo_id2, true)
            .await
            .unwrap();

        let repos = service
            .list_gitlab_provided_repositories_by_provider(vec![], None, None, None, None)
            .await
            .unwrap();

        assert_eq!(repos.len(), 1);
        assert!(repos[0].active);
    }

    #[tokio::test]
    async fn test_gitlab_repository_provider_crud() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = super::create(db.clone());

        let id = service
            .create_gitlab_repository_provider("id".into(), "secret".into())
            .await
            .unwrap();

        // Test retrieving gitlab provider by ID
        let provider1 = service
            .get_gitlab_repository_provider(id.clone())
            .await
            .unwrap();
        assert_eq!(
            provider1,
            GitlabRepositoryProvider {
                id: id.clone(),
                display_name: "id".into(),
                access_token: Some("secret".into()),
                connected: true,
            }
        );

        // Test listing gitlab providers
        let providers = service
            .list_gitlab_repository_providers(vec![], None, None, None, None)
            .await
            .unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].access_token, Some("secret".into()));

        // Test resetgitlab provider tokens
        service
            .reset_gitlab_repository_provider_access_token(id.clone())
            .await
            .unwrap();

        assert_eq!(
            service
                .get_gitlab_repository_provider(id.clone())
                .await
                .unwrap()
                .access_token,
            None
        );

        // Test deleting gitlab provider
        service
            .delete_gitlab_repository_provider(id.clone())
            .await
            .unwrap();

        assert_eq!(
            0,
            service
                .list_gitlab_repository_providers(vec![], None, None, None, None)
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
            .create_gitlab_provider("provider1".into(), "token".into())
            .await
            .unwrap();

        let repo_id = db
            .upsert_gitlab_provided_repository(
                provider_id,
                "vendor_id1".into(),
                "test_repo".into(),
                "https://gitlab.com/TabbyML/tabby".into(),
            )
            .await
            .unwrap();

        db.update_gitlab_provided_repository_active(repo_id, true)
            .await
            .unwrap();

        let git_urls = service.list_provided_git_urls().await.unwrap();
        assert_eq!(
            git_urls,
            ["https://token@gitlab.com/TabbyML/tabby".to_string()]
        );
    }

    #[tokio::test]
    async fn test_delete_outdated_repos() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());
        let time = Utc::now();

        let provider_id = db
            .create_gitlab_provider("provider1".into(), "secret1".into())
            .await
            .unwrap();

        let _repo_id = db
            .upsert_gitlab_provided_repository(
                provider_id,
                "vendor_id1".into(),
                "test_repo".into(),
                "https://gitlab.com/TabbyML/tabby".into(),
            )
            .await
            .unwrap();

        service
            .delete_outdated_gitlab_provided_repositories(provider_id.as_id(), time)
            .await
            .unwrap();

        assert_eq!(
            1,
            service
                .list_gitlab_provided_repositories_by_provider(vec![], None, None, None, None)
                .await
                .unwrap()
                .len()
        );

        let time = time + Duration::minutes(1);

        service
            .delete_outdated_gitlab_provided_repositories(provider_id.as_id(), time)
            .await
            .unwrap();

        assert_eq!(
            0,
            service
                .list_gitlab_provided_repositories_by_provider(vec![], None, None, None, None)
                .await
                .unwrap()
                .len()
        );
    }
}
