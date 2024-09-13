use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{graphql_object, GraphQLInputObject, GraphQLObject, ID};
use validator::{Validate,ValidationError};

use crate::{job::JobInfo, Result,CoreError};
use tabby_db::notion_documents:: NotionDocumentType;
use crate::juniper::relay;
use super::Context;

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct NotionDocument {
    pub name: String,
    pub id: ID,
    pub access_token: String,
    pub integration_id: String,
    pub integration_type: NotionDocumentType,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub job_info: JobInfo,
}

impl relay::NodeType for NotionDocument {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "NotionDocumentConnection"
    }

    fn edge_type_name() -> &'static str {
        "NotionocumentEdge"
    }
}

impl NotionDocument {
    pub fn source_id(&self) -> String {
        Self::format_source_id(&self.id)
    }

    pub fn format_source_id(id: &ID) -> String {
        format!("notion_document:{}", id)
    }
}



#[derive(Validate, GraphQLInputObject)]
pub struct CreateNotionDocumentInput {
    #[validate(regex(
        code = "name",
        path = "*crate::schema::constants::WEB_DOCUMENT_NAME_REGEX",
        message = "Invalid document name"
    ))]
    pub name: String,
    pub integration_id: String,
    #[validate(custom(function = "validate_notion_integration_type"))]
    pub integration_type: NotionDocumentType,
    pub access_token: String,
}

pub fn validate_notion_integration_type(integration_type: &NotionDocumentType) -> Result<(), ValidationError>{
    match integration_type {
        NotionDocumentType::Database => Ok(()),
        _ => Err(ValidationError::new("Currently only <Database> type is supported")),
    }
}


#[async_trait]
pub trait NotionDocumentService: Send + Sync {
    async fn list_notion_documents(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<NotionDocument>>;

    async fn create_notion_document(&self, input: CreateNotionDocumentInput) -> Result<ID>;
    async fn delete_notion_document(&self, id: ID) -> Result<bool>;
  
}

