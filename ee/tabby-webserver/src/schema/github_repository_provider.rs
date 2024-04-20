use async_trait::async_trait;
use juniper::{GraphQLInputObject, GraphQLObject, ID};
use lazy_static::lazy_static;
use regex::Regex;
use validator::Validate;

use super::Context;
use crate::{juniper::relay::NodeType, schema::Result};

lazy_static! {
    static ref GITHUB_REPOSITORY_PROVIDER_NAME_REGEX: Regex = Regex::new("^[\\w-]+$").unwrap();
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateGithubRepositoryProviderInput {
    #[validate(regex(code = "displayName", path = "GITHUB_REPOSITORY_PROVIDER_NAME_REGEX"))]
    pub display_name: String,
    #[validate(length(code = "applicationId", min = 20))]
    pub application_id: String,
    #[validate(length(code = "secret", min = 40))]
    pub secret: String,
}

#[derive(GraphQLInputObject, Validate)]
pub struct UpdateGithubRepositoryProviderInput {
    pub id: ID,
    #[validate(regex(code = "displayName", path = "GITHUB_REPOSITORY_PROVIDER_NAME_REGEX"))]
    pub display_name: String,
    #[validate(length(code = "applicationId", min = 20))]
    pub application_id: String,
    #[validate(length(code = "secret", min = 40))]
    pub secret: Option<String>,
}

#[derive(GraphQLObject, Debug, PartialEq)]
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
    async fn create_github_repository_provider(
        &self,
        display_name: String,
        application_id: String,
        application_secret: String,
    ) -> Result<ID>;
    async fn get_github_repository_provider(&self, id: ID) -> Result<GithubRepositoryProvider>;
    async fn delete_github_repository_provider(&self, id: ID) -> Result<()>;
    async fn update_github_repository_provider(
        &self,
        id: ID,
        display_name: String,
        application_id: String,
        secret: Option<String>,
    ) -> Result<()>;
    async fn read_github_repository_provider_secret(&self, id: ID) -> Result<String>;
    async fn update_github_repository_provider_access_token(
        &self,
        id: ID,
        access_token: String,
    ) -> Result<()>;

    async fn list_github_repository_providers(
        &self,
        ids: Vec<ID>,
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
    async fn list_provided_git_urls(&self) -> Result<Vec<String>>;
}
