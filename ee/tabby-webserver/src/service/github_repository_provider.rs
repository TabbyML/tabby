use crate::schema::Result;
use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;

use crate::schema::github_repository_provider::{
    GithubRepositoryProvider, GithubRepositoryProviderService,
};

use super::AsRowid;

struct GithubRepositoryProviderServiceImpl {
    db: DbConn,
}

pub fn new_github_repository_provider_service(db: DbConn) -> impl GithubRepositoryProviderService {
    GithubRepositoryProviderServiceImpl { db }
}

#[async_trait]
impl GithubRepositoryProviderService for GithubRepositoryProviderServiceImpl {
    async fn get_github_repository_provider(&self, id: ID) -> Result<GithubRepositoryProvider> {
        let provider = self.db.get_github_provider(id.as_rowid()? as i64).await?;
        Ok(GithubRepositoryProvider {
            display_name: provider.display_name,
            application_id: provider.application_id,
        })
    }

    async fn read_github_repository_provider_secret(&self, id: ID) -> Result<String> {
        let provider = self.db.get_github_provider(id.as_rowid()? as i64).await?;
        Ok(provider.secret)
    }

    async fn set_github_repository_provider_token(
        &self,
        id: ID,
        access_token: String,
    ) -> Result<()> {
        self.db
            .update_github_provider_token(id.as_rowid()? as i64, access_token)
            .await?;
        Ok(())
    }
}
