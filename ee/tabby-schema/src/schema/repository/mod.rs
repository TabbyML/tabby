mod types;
use std::{path::PathBuf, sync::Arc};

pub use types::*;

mod git;
pub use git::{CreateGitRepositoryInput, GitRepository, GitRepositoryService};

mod third_party;
use async_trait::async_trait;
use juniper::{GraphQLEnum, GraphQLObject, ID};
use serde::Deserialize;
use tabby_common::config::{RepositoryAccess, RepositoryConfig};
pub use third_party::{ProvidedRepository, ThirdPartyRepositoryService};

use super::Result;
use crate::{juniper::relay::NodeType, Context};

#[derive(GraphQLObject)]
pub struct FileEntrySearchResult {
    pub r#type: &'static str,
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

#[derive(GraphQLObject, Debug)]
pub struct Repository {
    pub id: ID,
    pub name: String,
    pub kind: RepositoryKind,

    #[graphql(skip)]
    pub dir: PathBuf,

    pub refs: Vec<String>,
}

impl From<GitRepository> for Repository {
    fn from(value: GitRepository) -> Self {
        let dir = RepositoryConfig::new(value.git_url).dir();
        let refs = tabby_search::GitReadOnly::list_refs(&dir).unwrap_or_default();
        Self {
            id: value.id,
            name: value.name,
            kind: RepositoryKind::Git,
            dir,
            refs,
        }
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

impl From<GithubProvidedRepository> for Repository {
    fn from(value: GithubProvidedRepository) -> Self {
        let dir = RepositoryConfig::new(value.git_url).dir();
        let refs = tabby_search::GitReadOnly::list_refs(&dir).unwrap_or_default();
        Self {
            id: value.id,
            name: value.name,
            kind: RepositoryKind::Github,
            dir,
            refs,
        }
    }
}

impl From<GitlabProvidedRepository> for Repository {
    fn from(value: GitlabProvidedRepository) -> Self {
        let dir = RepositoryConfig::new(value.git_url).dir();
        let refs = tabby_search::GitReadOnly::list_refs(&dir).unwrap_or_default();
        Self {
            id: value.id,
            name: value.name,
            kind: RepositoryKind::Gitlab,
            dir,
            refs,
        }
    }
}

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

#[derive(GraphQLObject, Debug, PartialEq)]
#[graphql(context = Context)]
pub struct GithubRepositoryProvider {
    pub id: ID,
    pub display_name: String,

    pub status: RepositoryProviderStatus,

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
    fn third_party(&self) -> Arc<dyn ThirdPartyRepositoryService>;
    fn access(self: Arc<Self>) -> Arc<dyn RepositoryAccess>;
}
