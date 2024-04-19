use std::sync::Arc;

use async_trait::async_trait;
use juniper::GraphQLObject;
use tabby_common::config::RepositoryAccess;

use super::{
    git_repository::GitRepositoryService,
    github_repository_provider::GithubRepositoryProviderService,
};

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

#[async_trait]
pub trait RepositoryService: Send + Sync + RepositoryAccess {
    fn git(&self) -> Arc<dyn GitRepositoryService>;
    fn github(&self) -> Arc<dyn GithubRepositoryProviderService>;
    fn repository_access(self: Arc<Self>) -> Arc<dyn RepositoryAccess>;
}
