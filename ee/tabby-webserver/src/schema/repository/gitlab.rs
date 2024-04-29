use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLObject, ID};

use super::{RepositoryProvider, RepositoryProviderStatus};
use crate::{
    juniper::relay::NodeType,
    schema::{Context, Result},
};

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
pub trait GitlabRepositoryService: Send + Sync + RepositoryProvider {
    async fn create_provider(&self, display_name: String, access_token: String) -> Result<ID>;
    async fn get_provider(&self, id: ID) -> Result<GitlabRepositoryProvider>;
    async fn delete_provider(&self, id: ID) -> Result<()>;
    async fn update_provider(
        &self,
        id: ID,
        display_name: String,
        access_token: String,
    ) -> Result<()>;
    async fn update_provider_status(&self, id: ID, success: bool) -> Result<()>;

    async fn list_providers(
        &self,
        ids: Vec<ID>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GitlabRepositoryProvider>>;

    async fn list_repositories(
        &self,
        provider: Vec<ID>,
        active: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GitlabProvidedRepository>>;

    async fn upsert_repository(
        &self,
        provider_id: ID,
        vendor_id: String,
        display_name: String,
        git_url: String,
    ) -> Result<()>;
    async fn update_repository_active(&self, id: ID, active: bool) -> Result<()>;
    async fn list_active_git_urls(&self) -> Result<Vec<String>>;
    async fn delete_outdated_repositories(
        &self,
        provider_id: ID,
        cutoff_timestamp: DateTime<Utc>,
    ) -> Result<()>;
}
