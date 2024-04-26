use async_trait::async_trait;
use juniper::{GraphQLObject, ID};
use validator::Validate;

use super::RepositoryProvider;
use crate::{
    juniper::relay::NodeType,
    schema::{Context, Result},
};

#[derive(Validate)]
pub struct CreateGitRepositoryInput {
    #[validate(regex(
        code = "name",
        path = "crate::schema::constants::REPOSITORY_NAME_REGEX",
        message = "Invalid repository name"
    ))]
    pub name: String,
    #[validate(url(code = "gitUrl", message = "Invalid Git URL"))]
    pub git_url: String,
}

#[derive(GraphQLObject, Debug)]
#[graphql(context = Context)]
pub struct GitRepository {
    pub id: juniper::ID,
    pub name: String,
    pub git_url: String,
}

impl NodeType for GitRepository {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "RepositoryConnection"
    }

    fn edge_type_name() -> &'static str {
        "RepositoryEdge"
    }
}

#[async_trait]
pub trait GitRepositoryService: Send + Sync + RepositoryProvider {
    async fn list(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GitRepository>>;

    async fn create(&self, name: String, git_url: String) -> Result<ID>;
    async fn delete(&self, id: &ID) -> Result<bool>;
    async fn update(&self, id: &ID, name: String, git_url: String) -> Result<bool>;
}
