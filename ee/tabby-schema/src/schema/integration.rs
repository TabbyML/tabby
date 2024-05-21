use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::ID;
use strum::EnumIter;

use crate::Result;

#[derive(Clone, EnumIter)]
pub enum IntegrationKind {
    Github,
    Gitlab,
    GithubSelfHosted,
    GitlabSelfHosted,
}

#[derive(PartialEq, Eq, Debug)]
pub enum IntegrationStatus {
    Ready,
    Pending,
    Failed,
}

pub struct Integration {
    pub id: ID,
    pub kind: IntegrationKind,
    pub display_name: String,
    pub access_token: String,
    pub api_base: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub status: IntegrationStatus,
}

impl Integration {
    pub fn api_base(&self) -> Result<&str> {
        match &self.kind {
            IntegrationKind::Github => Ok("https://api.github.com"),
            IntegrationKind::Gitlab => Ok("https://gitlab.com"),
            IntegrationKind::GithubSelfHosted => Ok(self
                .api_base
                .as_deref()
                .expect("Self-hosted github always has a specified api_base")),
            IntegrationKind::GitlabSelfHosted => Ok(self
                .api_base
                .as_deref()
                .expect("Self-hosted gitlab always has a specified api_base")),
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

    async fn get_integration(&self, id: ID) -> Result<Integration>;
    async fn update_integration_sync_status(&self, id: ID, error: Option<String>) -> Result<()>;
}
