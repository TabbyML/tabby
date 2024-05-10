mod types;
pub use types::*;

mod git;
pub use git::{CreateGitRepositoryInput, GitRepository, GitRepositoryService};

mod github;
pub use github::{GithubProvidedRepository, GithubRepositoryProvider, GithubRepositoryService};

mod gitlab;
use std::{path::PathBuf, sync::Arc};

mod third_party;
pub use third_party::{ProvidedRepository, ThirdPartyRepositoryService};

use async_trait::async_trait;
pub use gitlab::{GitlabProvidedRepository, GitlabRepositoryProvider, GitlabRepositoryService};
use juniper::{GraphQLEnum, GraphQLObject, ID};
use serde::Deserialize;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};

use super::Result;

#[derive(GraphQLObject, Debug)]
pub struct FileEntrySearchResult {
    pub r#type: String,
    pub path: String,

    /// matched indices for fuzzy search query.
    pub indices: Vec<i32>,
}

#[derive(GraphQLEnum, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RepositoryKind {
    Git,
    Github,
    Gitlab,
}

#[derive(GraphQLObject)]
pub struct Repository {
    pub id: ID,
    pub name: String,
    pub kind: RepositoryKind,

    #[graphql(skip)]
    pub dir: PathBuf,
}

impl From<GitRepository> for Repository {
    fn from(value: GitRepository) -> Self {
        Self {
            id: value.id,
            name: value.name,
            kind: RepositoryKind::Git,
            dir: RepositoryConfig::new(value.git_url).dir(),
        }
    }
}

impl From<GithubProvidedRepository> for Repository {
    fn from(value: GithubProvidedRepository) -> Self {
        Self {
            id: value.id,
            name: value.name,
            kind: RepositoryKind::Github,
            dir: RepositoryConfig::new(value.git_url).dir(),
        }
    }
}

impl From<GitlabProvidedRepository> for Repository {
    fn from(value: GitlabProvidedRepository) -> Self {
        Self {
            id: value.id,
            name: value.name,
            kind: RepositoryKind::Gitlab,
            dir: RepositoryConfig::new(value.git_url).dir(),
        }
    }
}

#[async_trait]
pub trait RepositoryProvider {
    async fn repository_list(&self) -> Result<Vec<Repository>>;
    async fn get_repository(&self, id: &ID) -> Result<Repository>;
}

#[async_trait]
pub trait RepositoryService: Send + Sync + RepositoryAccess {
    async fn repository_list(&self) -> Result<Vec<Repository>>;
    async fn resolve_repository(&self, kind: &RepositoryKind, id: &ID) -> Result<Repository>;
    async fn search_files(
        &self,
        kind: &RepositoryKind,
        id: &ID,
        pattern: &str,
        top_n: usize,
    ) -> Result<Vec<FileEntrySearchResult>>;

    fn git(&self) -> Arc<dyn GitRepositoryService>;
    fn github(&self) -> Arc<dyn GithubRepositoryService>;
    fn gitlab(&self) -> Arc<dyn GitlabRepositoryService>;
    fn access(self: Arc<Self>) -> Arc<dyn RepositoryAccess>;
}
