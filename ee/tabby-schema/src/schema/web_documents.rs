use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLInputObject, GraphQLObject, ID};
use validator::Validate;

use crate::{job::JobInfo, juniper::relay::NodeType, Context, Result};

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct CustomWebDocument {
    pub url: String,
    pub name: String,
    pub id: ID,
    pub created_at: DateTime<Utc>,
    pub job_info: JobInfo,
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct PresetWebDocument {
    pub name: String,
    pub id: ID,
    pub active: bool,
    pub job_info: Option<JobInfo>,
}

impl CustomWebDocument {
    pub fn source_id(&self) -> String {
        Self::format_source_id(&self.id)
    }

    pub fn format_source_id(id: &ID) -> String {
        format!("web_document:{}", id)
    }
}

#[derive(Validate, GraphQLInputObject)]
pub struct CreateCustomDocumentInput {
    #[validate(regex(
        code = "name",
        path = "*crate::schema::constants::USERNAME_REGEX",
        message = "Invalid repository name"
    ))]
    pub name: String,
    #[validate(url(code = "url", message = "Invalid URL"))]
    pub url: String,
}

#[derive(Validate, GraphQLInputObject)]
pub struct SetPresetDocumentActiveInput {
    #[validate(regex(
        code = "name",
        path = "*crate::schema::constants::USERNAME_REGEX",
        message = "Invalid repository name"
    ))]
    pub name: String,
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
        self.name.clone()
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
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<CustomWebDocument>>;

    async fn create_custom_web_document(&self, name: String, url: String) -> Result<ID>;
    async fn delete_custom_web_document(&self, id: ID) -> Result<()>;
    async fn list_preset_web_documents(&self,
                                       after: Option<String>,
                                       before: Option<String>,
                                       first: Option<usize>,
                                       last: Option<usize>,
                                       active: bool) -> Result<Vec<PresetWebDocument>>;
    async fn set_preset_web_documents_active(&self, name: String, active: bool) -> Result<ID>;
}
