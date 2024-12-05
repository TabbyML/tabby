use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{graphql_object, ID};
use tabby_common::config::CodeRepository;

use super::{GitReference, RepositoryProvider};
use crate::{
    context::ContextSourceIdValue, integration::IntegrationKind, job::JobInfo,
    juniper::relay::NodeType, schema::Result, Context,
};

pub struct ProvidedRepository {
    pub id: ID,
    pub integration_id: ID,
    pub active: bool,
    pub display_name: String,
    pub git_url: String,
    pub vendor_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub refs: Vec<GitReference>,

    pub job_info: JobInfo,
}

impl ProvidedRepository {
    pub fn format_source_id(id: &ID) -> String {
        format!("provided_repository:{}", id)
    }
}

#[graphql_object(context = Context, impl = [ContextSourceIdValue])]
impl ProvidedRepository {
    fn id(&self) -> &ID {
        &self.id
    }

    fn integration_id(&self) -> &ID {
        &self.integration_id
    }

    fn active(&self) -> bool {
        self.active
    }

    fn display_name(&self) -> &String {
        &self.display_name
    }

    fn git_url(&self) -> &String {
        &self.git_url
    }

    fn vendor_id(&self) -> &String {
        &self.vendor_id
    }

    fn created_at(&self) -> &DateTime<Utc> {
        &self.created_at
    }

    fn updated_at(&self) -> &DateTime<Utc> {
        &self.updated_at
    }

    fn refs(&self) -> &Vec<GitReference> {
        &self.refs
    }

    fn job_info(&self) -> &JobInfo {
        &self.job_info
    }

    pub fn source_id(&self) -> String {
        Self::format_source_id(&self.id)
    }
}

impl NodeType for ProvidedRepository {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "ProvidedRepositoryConnection"
    }

    fn edge_type_name() -> &'static str {
        "ProvidedRepositoryEdge"
    }
}

#[async_trait]
pub trait ThirdPartyRepositoryService: Send + Sync + RepositoryProvider {
    async fn list_repositories_with_filter(
        &self,
        integration_ids: Option<Vec<ID>>,
        kind: Option<IntegrationKind>,
        active: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<ProvidedRepository>>;

    async fn get_provided_repository(&self, id: ID) -> Result<ProvidedRepository>;

    async fn update_repository_active(&self, id: ID, active: bool) -> Result<()>;
    async fn upsert_repository(
        &self,
        integration_id: ID,
        vendor_id: String,
        display_name: String,
        git_url: String,
    ) -> Result<ID>;
    async fn sync_repositories(&self, integration_id: ID) -> Result<()>;
    async fn delete_outdated_repositories(
        &self,
        integration_id: ID,
        before: DateTime<Utc>,
    ) -> Result<usize>;
    async fn list_code_repositories(&self) -> Result<Vec<CodeRepository>>;
}
