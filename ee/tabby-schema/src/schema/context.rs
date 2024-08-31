use async_trait::async_trait;
use juniper::{GraphQLEnum, GraphQLObject, ID};

use super::{
    repository::{Repository, RepositoryKind},
    web_crawler::WebCrawlerUrl,
    web_documents::{CustomWebDocument, PresetWebDocument},
};
use crate::Result;

/// Represents the kind of context source.
#[derive(GraphQLEnum)]
pub enum ContextKind {
    Git,
    Github,
    Gitlab,
    Doc,
}

#[derive(GraphQLObject)]
pub struct ContextSource {
    pub id: ID,

    pub kind: ContextKind,

    /// Represents the source of the context, which is the value mapped to `source_id` in the index.
    pub source_id: String,

    /// Display name of the context, used to provide a human-readable name for user selection, such as in a dropdown menu.
    pub display_name: String,
}

impl ContextSource {
    pub fn is_code_repository(&self) -> bool {
        matches!(
            self.kind,
            ContextKind::Git | ContextKind::Github | ContextKind::Gitlab
        )
    }
}

impl From<Repository> for ContextSource {
    fn from(repo: Repository) -> Self {
        let kind = match repo.kind {
            RepositoryKind::Git | RepositoryKind::GitConfig => ContextKind::Git,
            RepositoryKind::Github | RepositoryKind::GithubSelfHosted => ContextKind::Github,
            RepositoryKind::Gitlab | RepositoryKind::GitlabSelfHosted => ContextKind::Gitlab,
        };

        Self {
            id: ID::from(repo.source_id.clone()),
            kind,
            source_id: repo.source_id,
            display_name: repo.git_url,
        }
    }
}

impl From<WebCrawlerUrl> for ContextSource {
    fn from(url: WebCrawlerUrl) -> Self {
        Self {
            id: ID::from(url.source_id()),
            kind: ContextKind::Doc,
            source_id: url.source_id(),
            display_name: url.url,
        }
    }
}

impl From<PresetWebDocument> for ContextSource {
    fn from(doc: PresetWebDocument) -> Self {
        Self {
            id: ID::from(doc.source_id()),
            kind: ContextKind::Doc,
            source_id: doc.source_id(),
            display_name: doc.name,
        }
    }
}

impl From<CustomWebDocument> for ContextSource {
    fn from(doc: CustomWebDocument) -> Self {
        Self {
            id: ID::from(doc.source_id()),
            kind: ContextKind::Doc,
            source_id: doc.source_id(),
            display_name: doc.name,
        }
    }
}

#[derive(GraphQLObject)]
pub struct ContextInfo {
    pub sources: Vec<ContextSource>,

    /// Whether the deployment has capability to search public web.
    pub can_search_public: bool,
}

#[async_trait]
pub trait ContextService: Send + Sync {
    async fn read(&self) -> Result<ContextInfo>;
}
