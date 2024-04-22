use std::sync::Arc;

use async_trait::async_trait;
use juniper::GraphQLObject;
use tabby_common::config::RepositoryAccess;
use tabby_search::FileSearch;

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

impl From<FileSearch> for FileEntrySearchResult {
    fn from(value: FileSearch) -> Self {
        Self {
            r#type: value.r#type,
            path: value.path,
            indices: value.indices,
        }
    }
}

#[async_trait]
pub trait RepositoryService: Send + Sync + RepositoryAccess {
    fn git(&self) -> Arc<dyn GitRepositoryService>;
    fn github(&self) -> Arc<dyn GithubRepositoryProviderService>;
    fn access(self: Arc<Self>) -> Arc<dyn RepositoryAccess>;
}
