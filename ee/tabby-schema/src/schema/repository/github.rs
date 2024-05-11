use async_trait::async_trait;
use juniper::{GraphQLObject, ID};

use super::{RepositoryProvider, RepositoryProviderStatus};
use crate::{
    juniper::relay::NodeType,
    schema::{Context, Result},
};

#[derive(GraphQLObject, Debug, PartialEq)]
#[graphql(context = Context)]
pub struct GithubRepositoryProvider {
    pub id: ID,
    pub display_name: String,

    pub status: RepositoryProviderStatus,

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
pub trait GithubRepositoryService: Send + Sync + RepositoryProvider {
    async fn create_provider(&self, display_name: String, access_token: String) -> Result<ID>;
    async fn delete_provider(&self, id: ID) -> Result<()>;
    async fn update_provider(
        &self,
        id: ID,
        display_name: String,
        access_token: Option<String>,
    ) -> Result<()>;

    async fn list_providers(
        &self,
        ids: Vec<ID>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GithubRepositoryProvider>>;

    async fn list_repositories(
        &self,
        provider: Vec<ID>,
        active: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GithubProvidedRepository>>;

    async fn update_repository_active(&self, id: ID, active: bool) -> Result<()>;
    async fn list_active_git_urls(&self) -> Result<Vec<String>>;
}
