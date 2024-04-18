use async_trait::async_trait;
use juniper::{GraphQLObject, ID};
use lazy_static::lazy_static;
use regex::Regex;
use validator::Validate;

use super::{Context, Result};
use crate::juniper::relay::NodeType;

lazy_static! {
    static ref REPOSITORY_NAME_REGEX: Regex = Regex::new("").unwrap();
}

#[derive(Validate)]
pub struct CreateGitRepositoryInput {
    #[validate(regex(
        code = "name",
        path = "self::REPOSITORY_NAME_REGEX",
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

#[derive(GraphQLObject, Debug)]
pub struct FileEntrySearchResult {
    pub r#type: String,
    pub path: String,

    /// matched indices for fuzzy search query.
    pub indices: Vec<i32>,
}

impl FileEntrySearchResult {
    pub fn new(r#type: String, path: String, indices: Vec<u32>) -> Self {
        Self {
            r#type,
            path,
            indices: indices.into_iter().map(|i| i as i32).collect(),
        }
    }
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
pub trait GitRepositoryService: Send + Sync {
    async fn list(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<GitRepository>>;

    async fn create(&self, name: String, git_url: String) -> Result<ID>;
    async fn get_by_name(&self, name: &str) -> Result<GitRepository>;
    async fn delete(&self, id: &ID) -> Result<bool>;
    async fn update(&self, id: &ID, name: String, git_url: String) -> Result<bool>;

    async fn search_files(
        &self,
        name: &str,
        pattern: &str,
        top_n: usize,
    ) -> Result<Vec<FileEntrySearchResult>>;
}
