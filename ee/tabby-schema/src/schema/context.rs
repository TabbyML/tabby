use std::collections::HashMap;

use async_trait::async_trait;
use juniper::{GraphQLEnum, GraphQLObject, ID};
use regex::{Captures, Regex};

use super::{
    repository::{Repository, RepositoryKind},
    web_documents::{CustomWebDocument, PresetWebDocument},
};
use crate::{policy::AccessPolicy, Result};

/// Represents the kind of context source.
#[derive(GraphQLEnum)]
pub enum ContextKind {
    Git,
    Github,
    Gitlab,
    Doc,
    Web,
}

pub const PUBLIC_WEB_INTERNAL_SOURCE_ID: &str = "internal-public-web";

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

        let display_name = match repo.kind {
            RepositoryKind::Git
            | RepositoryKind::GitConfig
            | RepositoryKind::GithubSelfHosted
            | RepositoryKind::GitlabSelfHosted => repo.git_url,
            RepositoryKind::Github | RepositoryKind::Gitlab => repo.name,
        };

        Self {
            id: ID::from(repo.source_id.clone()),
            kind,
            source_id: repo.source_id,
            display_name,
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
}

impl ContextInfo {
    pub fn helper(&self) -> ContextInfoHelper {
        ContextInfoHelper::new(self)
    }
}

pub struct ContextInfoHelper<'k, 'v> {
    sources: HashMap<&'k str, &'v str>,
}

impl<'a> ContextInfoHelper<'a, 'a> {
    pub fn new(context_info: &'a ContextInfo) -> Self {
        Self {
            sources: context_info
                .sources
                .iter()
                .map(|source| (source.source_id.as_str(), source.display_name.as_str()))
                .collect(),
        }
    }

    /// Replace content tagged with `[[source:${id}]]` with its display name.
    pub fn rewrite_tag(&self, content: &str) -> String {
        let re = Regex::new(r"\[\[source:(.*?)\]\]").unwrap();
        let new_content = re.replace_all(content, |caps: &Captures| {
            let source_id = caps.get(1).unwrap().as_str();
            if source_id == PUBLIC_WEB_INTERNAL_SOURCE_ID {
                // For public-web source, don't include it in the content.
                return "".to_owned();
            }
            if let Some(display_name) = self.sources.get(source_id) {
                display_name.to_string()
            } else {
                caps[0].to_owned()
            }
        });
        new_content.to_string()
    }

    pub fn can_access_source_id(&self, source_id: &str) -> bool {
        self.sources.contains_key(source_id)
    }
}

#[async_trait]
pub trait ContextService: Send + Sync {
    /// Read context information from the backend. If `policy` is `None`, this retrieves all context information without applying any access policy.
    async fn read(&self, policy: Option<&AccessPolicy>) -> Result<ContextInfo>;
}
