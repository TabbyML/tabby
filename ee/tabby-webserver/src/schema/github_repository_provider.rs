use anyhow::Result;
use async_trait::async_trait;
use juniper::{GraphQLObject, ID};

#[derive(GraphQLObject)]
pub struct GithubRepositoryProvider {
    pub display_name: String,
    pub application_id: String,
}

#[async_trait]
pub trait GithubRepositoryProviderService: Send + Sync {
    async fn get_github_repository_provider(&self, id: ID) -> Result<GithubRepositoryProvider>;
    async fn read_github_repository_provider_secret(&self, id: ID) -> Result<String>;
    async fn set_github_repository_provider_token(
        &self,
        id: ID,
        access_token: String,
    ) -> Result<()>;
}
