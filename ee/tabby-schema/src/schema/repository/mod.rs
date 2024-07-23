mod types;
use std::{path::PathBuf, sync::Arc};

pub use types::*;

mod git;
pub use git::{CreateGitRepositoryInput, GitRepository, GitRepositoryService};

mod third_party;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use juniper::{graphql_object, GraphQLEnum, GraphQLObject, ID};
use serde::Deserialize;
use tabby_common::config::RepositoryConfig;
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

#[derive(GraphQLEnum, Debug, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RepositoryKind {
    Git,
    Github,
    Gitlab,
    GithubSelfHosted,
    GitlabSelfHosted,
}

#[derive(GraphQLObject, Debug)]
pub struct Repository {
    pub id: ID,
    pub name: String,
    pub kind: RepositoryKind,

    #[graphql(skip)]
    pub dir: PathBuf,

    pub git_url: String,
    pub refs: Vec<String>,
}

impl From<GitRepository> for Repository {
    fn from(value: GitRepository) -> Self {
        let config = RepositoryConfig::new(value.git_url);
        Self {
            id: value.id,
            name: value.name,
            kind: RepositoryKind::Git,
            dir: config.dir(),
            git_url: config.canonical_git_url(),
            refs: value.refs,
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
    pub refs: Vec<String>,
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
    pub refs: Vec<String>,
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
        let config = RepositoryConfig::new(value.git_url);
        Self {
            id: value.id,
            name: value.name,
            kind: RepositoryKind::Github,
            dir: config.dir(),
            git_url: config.canonical_git_url(),
            refs: value.refs,
        }
    }
}

impl From<GitlabProvidedRepository> for Repository {
    fn from(value: GitlabProvidedRepository) -> Self {
        let config = RepositoryConfig::new(value.git_url);
        Self {
            id: value.id,
            name: value.name,
            kind: RepositoryKind::Gitlab,
            dir: config.dir(),
            git_url: config.canonical_git_url(),
            refs: value.refs,
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

    pub api_base: Option<String>,
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

    pub api_base: Option<String>,
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

#[derive(GraphQLObject)]
pub struct GrepFile {
    pub path: String,
    pub lines: Vec<GrepLine>,
}

#[derive(GraphQLObject)]
pub struct GrepLine {
    /// Content of the line.
    pub line: GrepTextOrBase64,

    /// Byte offset in the file to the start of the line.
    pub byte_offset: i32,

    /// Line number in the file, starting from 1.
    pub line_number: i32,

    /// The matches in the line.
    pub sub_matches: Vec<GrepSubMatch>,
}

pub enum GrepTextOrBase64 {
    Text(String),
    Base64(Vec<u8>),
}

#[graphql_object]
impl GrepTextOrBase64 {
    fn text(&self) -> Option<&str> {
        match self {
            GrepTextOrBase64::Text(text) => Some(text),
            _ => None,
        }
    }

    fn base64(&self) -> Option<String> {
        match self {
            GrepTextOrBase64::Base64(bytes) => Some(STANDARD.encode(bytes)),
            _ => None,
        }
    }
}

#[derive(GraphQLObject)]
pub struct GrepSubMatch {
    // Byte offsets in the line
    pub bytes_start: i32,
    pub bytes_end: i32,
}

#[async_trait]
pub trait RepositoryProvider {
    async fn repository_list(&self) -> Result<Vec<Repository>>;
    async fn get_repository(&self, id: &ID) -> Result<Repository>;
}

#[async_trait]
pub trait RepositoryService: Send + Sync {
    async fn repository_list(&self) -> Result<Vec<Repository>>;
    async fn resolve_repository(&self, kind: &RepositoryKind, id: &ID) -> Result<Repository>;
    async fn search_files(
        &self,
        kind: &RepositoryKind,
        id: &ID,
        rev: Option<&str>,
        pattern: &str,
        top_n: usize,
    ) -> Result<Vec<FileEntrySearchResult>>;

    async fn grep(
        &self,
        kind: &RepositoryKind,
        id: &ID,
        rev: Option<&str>,
        query: &str,
        top_n: usize,
    ) -> Result<Vec<GrepFile>>;

    fn git(&self) -> Arc<dyn GitRepositoryService>;
    fn third_party(&self) -> Arc<dyn ThirdPartyRepositoryService>;

    async fn list_all_repository_urls(&self) -> Result<Vec<RepositoryConfig>>;
    async fn list_all_sources(&self) -> Result<Vec<(String, String)>>;

    async fn resolve_web_source_id_by_git_url(&self, git_url: &str) -> Result<String>;
}
