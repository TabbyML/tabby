use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;

use super::{AsID, AsRowid};
use crate::{
    schema::{
        github_repository_provider::{GithubRepositoryProvider, GithubRepositoryProviderService},
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
        access_token: String,
    ) -> Result<()> {
        self.db
            .update_github_provider_access_token(id.as_rowid()?, access_token)
            .await?;
        Ok(())
    }

    async fn list_github_repository_providers(
        &self,
        ids: Option<Vec<i32>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GithubRepositoryProvider>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let providers = self
            .db
            .list_github_repository_providers(ids, limit, skip_id, backwards)
            .await?;
        Ok(providers
            .into_iter()
            .map(GithubRepositoryProvider::from)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_github_repository_provider_crud() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = new_github_repository_provider_service(db);

        let id = service
            .create_github_repository_provider("example".into(), "id".into(), "secret".into())
            .await
            .unwrap();

        let provider1 = service
            .get_github_repository_provider(id.clone())
            .await
            .unwrap();
        assert_eq!(
            provider1,
            GithubRepositoryProvider {
                id: id.clone(),
                display_name: "example".into(),
                application_id: "id".into()
            }
        );

        let secret1 = service
            .read_github_repository_provider_secret(id.clone())
            .await
            .unwrap();
        assert_eq!(secret1, "secret");

        assert_eq!(
            1,
            service
                .list_github_repository_providers(None, None, None, None, None)
                .await
                .unwrap()
                .len()
        );

        service
            .delete_github_repository_provider(id.clone())
            .await
            .unwrap();

        assert_eq!(
            0,
            service
                .list_github_repository_providers(None, None, None, None, None)
                .await
                .unwrap()
                .len()
        );
    }
}
