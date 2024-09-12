use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{graphql_object, GraphQLInputObject, ID};
use validator::Validate;

use super::context::{ContextSourceIdValue, ContextSourceKind, ContextSourceValue};
use crate::{job::JobInfo, juniper::relay::NodeType, Context, Result};

pub struct CustomWebDocument {
    pub url: String,
    pub name: String,
    pub id: ID,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub job_info: JobInfo,
}

impl CustomWebDocument {
    pub fn format_source_id(id: &ID) -> String {
        format!("custom_web_document:{}", id)
    }
}

#[graphql_object(context = Context, impl = [ContextSourceIdValue, ContextSourceValue])]
impl CustomWebDocument {
    fn url(&self) -> &str {
        &self.url
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> &ID {
        &self.id
    }

    fn created_at(&self) -> &DateTime<Utc> {
        &self.created_at
    }

    fn updated_at(&self) -> &DateTime<Utc> {
        &self.updated_at
    }

    fn job_info(&self) -> &JobInfo {
        &self.job_info
    }

    fn source_kind(&self) -> ContextSourceKind {
        ContextSourceKind::Doc
    }

    pub fn source_id(&self) -> String {
        Self::format_source_id(&self.id)
    }

    pub fn source_name(&self) -> &str {
        &self.name
    }
}

pub struct PresetWebDocument {
    pub id: ID,

    pub name: String,
    pub updated_at: Option<DateTime<Utc>>,
    pub job_info: Option<JobInfo>,
    pub is_active: bool,
}

impl PresetWebDocument {
    pub fn format_source_id(name: &String) -> String {
        format!("preset_web_document:{}", name)
    }
}

#[graphql_object(context = Context, impl = [ContextSourceIdValue, ContextSourceValue])]
impl PresetWebDocument {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> &ID {
        &self.id
    }

    /// `updated_at` is only filled when the preset is active.
    fn updated_at(&self) -> &Option<DateTime<Utc>> {
        &self.updated_at
    }

    /// `job_info` is only filled when the preset is active.
    fn job_info(&self) -> &Option<JobInfo> {
        &self.job_info
    }

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn source_kind(&self) -> ContextSourceKind {
        ContextSourceKind::Doc
    }

    pub fn source_id(&self) -> String {
        Self::format_source_id(&self.name)
    }

    pub fn source_name(&self) -> &str {
        &self.name
    }
}

#[derive(Validate, GraphQLInputObject)]
pub struct CreateCustomDocumentInput {
    #[validate(regex(
        code = "name",
        path = "*crate::schema::constants::WEB_DOCUMENT_NAME_REGEX",
        message = "Invalid document name"
    ))]
    pub name: String,
    #[validate(url(code = "url", message = "Invalid URL"))]
    pub url: String,
}

#[derive(Validate, GraphQLInputObject)]
pub struct SetPresetDocumentActiveInput {
    pub id: ID,
    pub active: bool,
}

impl NodeType for CustomWebDocument {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "CustomDocumentConnection"
    }

    fn edge_type_name() -> &'static str {
        "CustomDocumentEdge"
    }
}

impl NodeType for PresetWebDocument {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "PresetDocumentConnection"
    }

    fn edge_type_name() -> &'static str {
        "PresetDocumentEdge"
    }
}

#[async_trait]
pub trait WebDocumentService: Send + Sync {
    async fn list_custom_web_documents(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<CustomWebDocument>>;

    async fn create_custom_web_document(&self, name: String, url: String) -> Result<ID>;
    async fn delete_custom_web_document(&self, id: ID) -> Result<()>;
    async fn list_preset_web_documents(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
        is_active: Option<bool>,
    ) -> Result<Vec<PresetWebDocument>>;
    async fn set_preset_web_documents_active(&self, id: ID, active: bool) -> Result<()>;
}
