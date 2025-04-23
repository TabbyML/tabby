use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject, ID};
use strum::EnumIter;
use url::Url;

use crate::{juniper::relay::NodeType, Context, CoreError, Result};

#[derive(Clone, EnumIter, GraphQLEnum)]
pub enum IntegrationKind {
    Github,
    Gitlab,
    GithubSelfHosted,
    GitlabSelfHosted,
}

impl IntegrationKind {
    pub fn format_authenticated_url(&self, git_url: &str, access_token: &str) -> Result<String> {
        let mut url = Url::parse(git_url).map_err(|e| CoreError::Other(e.into()))?;
        match self {
            IntegrationKind::Github | IntegrationKind::GithubSelfHosted => {
                let _ = url.set_username(access_token);
            }
            IntegrationKind::Gitlab | IntegrationKind::GitlabSelfHosted => {
                let _ = url.set_username("oauth2");
                let _ = url.set_password(Some(access_token));
            }
        }
        Ok(url.to_string())
    }

    pub fn is_self_hosted(&self) -> bool {
        match self {
            IntegrationKind::Github => false,
            IntegrationKind::Gitlab => false,
            IntegrationKind::GithubSelfHosted => true,
            IntegrationKind::GitlabSelfHosted => true,
        }
    }
}

#[derive(PartialEq, Eq, Debug, GraphQLEnum)]
pub enum IntegrationStatus {
    Ready,
    Pending,
    Failed,
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct Integration {
    pub id: ID,
    pub kind: IntegrationKind,
    pub display_name: String,
    pub access_token: String,
    pub api_base: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub status: IntegrationStatus,
    pub message: Option<String>,
}

impl NodeType for Integration {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "IntegrationConnection"
    }

    fn edge_type_name() -> &'static str {
        "IntegrationEdge"
    }
}

impl Integration {
    pub fn api_base(&self) -> &str {
        match &self.kind {
            IntegrationKind::Github => "https://api.github.com",
            IntegrationKind::Gitlab => "https://gitlab.com",
            IntegrationKind::GithubSelfHosted => self
                .api_base
                .as_deref()
                .expect("Self-hosted github always has a specified api_base"),
            IntegrationKind::GitlabSelfHosted => self
                .api_base
                .as_deref()
                .expect("Self-hosted gitlab always has a specified api_base"),
        }
    }
}

#[async_trait]
pub trait IntegrationService: Send + Sync {
    async fn create_integration(
        &self,
        kind: IntegrationKind,
        display_name: String,
        access_token: String,
        api_base: Option<String>,
    ) -> Result<ID>;

    async fn delete_integration(&self, id: ID, kind: IntegrationKind) -> Result<()>;

    async fn update_integration(
        &self,
        id: ID,
        kind: IntegrationKind,
        display_name: String,
        access_token: Option<String>,
        api_base: Option<String>,
    ) -> Result<()>;

    async fn list_integrations(
        &self,
        ids: Option<Vec<ID>>,
        kind: Option<IntegrationKind>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Integration>>;

    async fn get_integration(&self, id: &ID) -> Result<Integration>;
    async fn update_integration_sync_status(&self, id: &ID, error: Option<String>) -> Result<()>;
}
