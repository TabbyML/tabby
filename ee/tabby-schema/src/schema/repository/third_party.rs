use crate::{integration::IntegrationKind, schema::Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::ID;

pub struct ProvidedRepository {
    pub integration_id: ID,
    pub active: bool,
    pub kind: IntegrationKind,
    pub display_name: String,
    pub git_url: String,
    pub vendor_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[async_trait]
pub trait ThirdPartyRepositoryService: Send + Sync {
    async fn list_repositories(
        &self,
        integration_ids: Option<Vec<ID>>,
        kind: Option<IntegrationKind>,
        active: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<ProvidedRepository>>;

    async fn update_repository_active(&self, id: ID, active: bool) -> Result<()>;
    async fn list_active_git_urls(&self) -> Result<Vec<String>>;
}
