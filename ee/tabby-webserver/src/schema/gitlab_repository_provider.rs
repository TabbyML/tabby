use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLObject, ID};

use super::{repository::RepositoryProvider, types::RepositoryProviderStatus, Context};
use crate::{juniper::relay::NodeType, schema::Result};

#[derive(GraphQLObject, Debug, PartialEq)]
#[graphql(context = Context)]
pub struct GitlabRepositoryProvider {
    pub id: ID,
    pub display_name: String,

    pub status: RepositoryProviderStatus,

    #[graphql(skip)]
    pub access_token: Option<String>,
}

impl NodeType for GitlabRepositoryProvider {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "GitlabRepositoryProviderConnection"
    }

    fn edge_type_name() -> &'static str {
        "GitlabRepositoryProviderEdge"
    }
}

#[derive(GraphQLObject, Debug)]
#[graphql(context = Context)]
pub struct GitlabProvidedRepository {
    pub id: ID,
    pub vendor_id: String,
    pub gitlab_repository_provider_id: ID,
    pub name: String,
    pub git_url: String,
    pub active: bool,
}

impl NodeType for GitlabProvidedRepository {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "GitlabProvidedRepositoryConnection"
    }

    fn edge_type_name() -> &'static str {
        "GitlabProvidedRepositoryEdge"
    }
}

#[async_trait]
pub trait GitlabRepositoryProviderService: Send + Sync + RepositoryProvider {
    async fn create_gitlab_repository_provider(
        &self,
        display_name: String,
        access_token: String,
    ) -> Result<ID>;
    async fn get_gitlab_repository_provider(&self, id: ID) -> Result<GitlabRepositoryProvider>;
    async fn delete_gitlab_repository_provider(&self, id: ID) -> Result<()>;
    async fn update_gitlab_repository_provider(
        &self,
        id: ID,
        display_name: String,
        access_token: String,
    ) -> Result<()>;
    async fn update_gitlab_repository_provider_sync_status(
        &self,
        id: ID,
        success: bool,
    ) -> Result<()>;

    async fn list_gitlab_repository_providers(
        &self,
        ids: Vec<ID>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GitlabRepositoryProvider>>;

    async fn list_gitlab_provided_repositories_by_provider(
        &self,
        provider: Vec<ID>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GitlabProvidedRepository>>;

    async fn upsert_gitlab_provided_repository(
        &self,
        provider_id: ID,
        vendor_id: String,
        display_name: String,
        git_url: String,
    ) -> Result<()>;
    async fn update_gitlab_provided_repository_active(&self, id: ID, active: bool) -> Result<()>;
    async fn list_provided_git_urls(&self) -> Result<Vec<String>>;
    async fn delete_outdated_gitlab_provided_repositories(
        &self,
        provider_id: ID,
        cutoff_timestamp: DateTime<Utc>,
    ) -> Result<()>;
}
