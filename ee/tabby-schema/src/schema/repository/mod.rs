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
use tabby_common::config::{CodeRepository, RepositoryConfig};
pub use third_party::{ProvidedRepository, ThirdPartyRepositoryService};

use super::{
    context::{ContextSourceIdValue, ContextSourceKind, ContextSourceValue},
    Result,
};
use crate::{juniper::relay::NodeType, policy::AccessPolicy, Context};

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
    GitConfig,
}

#[derive(Debug)]
pub struct Repository {
    pub id: ID,

    pub source_id: String,

    pub name: String,
    pub kind: RepositoryKind,

    pub dir: PathBuf,

    pub git_url: String,
    pub refs: Vec<GitReference>,
}

#[graphql_object(context = Context, impl = [ContextSourceIdValue, ContextSourceValue])]
impl Repository {
    fn id(&self) -> &ID {
        &self.id
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    fn source_kind(&self) -> ContextSourceKind {
        match self.kind {
            RepositoryKind::Git | RepositoryKind::GitConfig => ContextSourceKind::Git,
            RepositoryKind::Github | RepositoryKind::GithubSelfHosted => ContextSourceKind::Github,
            RepositoryKind::Gitlab | RepositoryKind::GitlabSelfHosted => ContextSourceKind::Gitlab,
        }
    }

    pub fn source_name(&self) -> &str {
        match self.kind {
            RepositoryKind::Git
            | RepositoryKind::GitConfig
            | RepositoryKind::GithubSelfHosted
            | RepositoryKind::GitlabSelfHosted => &self.git_url,
            RepositoryKind::Github | RepositoryKind::Gitlab => &self.name,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> RepositoryKind {
        self.kind
    }

    fn git_url(&self) -> &str {
        &self.git_url
    }

    fn refs(&self) -> &[GitReference] {
        &self.refs
    }
}

#[derive(GraphQLObject, Debug)]
pub struct GitReference {
    pub name: String,
    pub commit: String,
}

impl From<GitRepository> for Repository {
    fn from(value: GitRepository) -> Self {
        Self {
            source_id: value.source_id(),
            id: ID::new(value.source_id()),
            name: value.name,
            kind: RepositoryKind::Git,
            dir: RepositoryConfig::resolve_dir(&value.git_url),
            git_url: RepositoryConfig::canonicalize_url(&value.git_url),
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
pub struct RepositoryGrepOutput {
    pub files: Vec<GrepFile>,

    /// Elapsed time in milliseconds for grep search.
    pub elapsed_ms: i32,
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
    /// Read repositories. If `policy` is `None`, this retrieves all repositories without applying any access policy.
    async fn repository_list(&self, policy: Option<&AccessPolicy>) -> Result<Vec<Repository>>;
    async fn resolve_repository(
        &self,
        policy: &AccessPolicy,
        kind: &RepositoryKind,
        id: &ID,
    ) -> Result<Repository>;

    async fn search_files(
        &self,
        policy: &AccessPolicy,
        kind: &RepositoryKind,
        id: &ID,
        rev: Option<&str>,
        pattern: &str,
        top_n: usize,
    ) -> Result<Vec<FileEntrySearchResult>>;

    async fn grep(
        &self,
        policy: &AccessPolicy,
        kind: &RepositoryKind,
        id: &ID,
        rev: Option<&str>,
        query: &str,
        top_n: usize,
    ) -> Result<Vec<GrepFile>>;

    fn git(&self) -> Arc<dyn GitRepositoryService>;
    fn third_party(&self) -> Arc<dyn ThirdPartyRepositoryService>;

    async fn list_all_code_repository(&self) -> Result<Vec<CodeRepository>>;

    async fn resolve_source_id_by_git_url(&self, git_url: &str) -> Result<String>;
}
