use anyhow::Result;
use async_trait::async_trait;
use juniper::{FieldError, GraphQLObject, IntoFieldError, ScalarValue, ID};
use juniper_axum::relay::NodeType;
use validator::{Validate, ValidationErrors};

use super::{from_validation_errors, Context};

#[derive(Validate)]
pub struct CreateRepositoryInput {
    #[validate(regex = "tabby_common::config::REPOSITORY_NAME_REGEX")]
    pub name: String,
    #[validate(url)]
    pub git_url: String,
}

#[derive(thiserror::Error, Debug)]
pub enum RepositoryError {
    #[error("Invalid input parameters")]
    InvalidInput(#[from] ValidationErrors),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl<S: ScalarValue> IntoFieldError<S> for RepositoryError {
    fn into_field_error(self) -> FieldError<S> {
        match self {
            Self::InvalidInput(errors) => from_validation_errors(errors),
            _ => self.into(),
        }
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
