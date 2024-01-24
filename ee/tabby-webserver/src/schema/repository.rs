use anyhow::Result;
use async_trait::async_trait;
use juniper::{FieldError, GraphQLObject, IntoFieldError, ScalarValue, ID};
use juniper_axum::relay::NodeType;

use super::Context;

#[derive(thiserror::Error, Debug)]
pub enum RepositoryError {
    #[error("Invalid repository name")]
    InvalidName,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl<S: ScalarValue> IntoFieldError<S> for RepositoryError {
    fn into_field_error(self) -> FieldError<S> {
        self.into()
    }
}

#[derive(GraphQLObject, Debug)]
#[graphql(context = Context)]
pub struct Repository {
    pub id: juniper::ID,
    pub name: String,
    pub git_url: String,
}

impl NodeType for Repository {
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
pub trait RepositoryService: Send + Sync {
    async fn list_repositories(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Repository>>;

    async fn create_repository(&self, name: String, git_url: String) -> Result<ID>;
    async fn delete_repository(&self, id: ID) -> Result<bool>;
    async fn update_repository(&self, id: ID, name: String, git_url: String) -> Result<bool>;
}
