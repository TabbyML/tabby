use async_trait::async_trait;
use juniper::{GraphQLObject, ID};

use super::Context;
use crate::{juniper::relay::NodeType, schema::Result};

#[derive(GraphQLObject, Debug)]
#[graphql(context = Context)]
pub struct GithubRepositoryProvider {
    pub id: ID,
    pub display_name: String,
    pub application_id: String,
    #[graphql(skip)]
    pub secret: String,
    #[graphql(skip)]
    pub access_token: Option<String>,
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
    pub vendor_id: String,
    pub github_repository_provider_id: ID,
    pub name: String,
    pub git_url: String,
    pub active: bool,
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
    async fn get_github_repository_provider(&self, id: ID) -> Result<GithubRepositoryProvider>;
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
        provider: Vec<ID>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GithubProvidedRepository>>;

    async fn update_github_provided_repository_active(&self, id: ID, active: bool) -> Result<()>;
}
