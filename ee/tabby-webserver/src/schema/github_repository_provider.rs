use async_trait::async_trait;
use juniper::{GraphQLObject, ID};
use juniper_axum::relay::NodeType;

use super::Context;
use crate::schema::Result;

#[derive(GraphQLObject, Debug)]
#[graphql(context = Context)]
pub struct GithubRepositoryProvider {
    pub id: ID,
    pub display_name: String,
    pub application_id: String,
}

impl NodeType for GithubRepositoryProvider {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "GithubRepositoryProviderConnection"
    }

    fn edge_type_name() -> &'static str {
        "GithubRepositoryProviderEdge"
    }
}

#[derive(GraphQLObject, Debug)]
#[graphql(context = Context)]
pub struct GithubProvidedRepository {
    pub id: ID,
    pub github_repository_provider_id: ID,
    pub name: String,
    pub git_url: String,
}

impl NodeType for GithubProvidedRepository {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "GithubProvidedRepositoryConnection"
    }

    fn edge_type_name() -> &'static str {
        "GithubProvidedRepositoryEdge"
    }
}

#[async_trait]
pub trait GithubRepositoryProviderService: Send + Sync {
    async fn create_github_repository_provider(
        &self,
        display_name: String,
        application_id: String,
        application_secret: String,
    ) -> Result<ID>;
    async fn get_github_repository_provider(&self, id: ID) -> Result<GithubRepositoryProvider>;
    async fn read_github_repository_provider_secret(&self, id: ID) -> Result<String>;
    async fn update_github_repository_provider_access_token(
        &self,
        id: ID,
        access_token: String,
    ) -> Result<()>;

    async fn list_github_repository_providers(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GithubRepositoryProvider>>;

    async fn list_github_provided_repositories_by_provider(
        &self,
        provider: ID,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GithubProvidedRepository>>;
}
