use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLInputObject, GraphQLObject, ID};
use validator::Validate;

use crate::{job::JobInfo, juniper::relay::NodeType, Context, Result};

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct WebDocument {
    pub url: String,
    pub id: ID,
    pub created_at: DateTime<Utc>,
    pub job_info: JobInfo,
}

impl WebDocument {
    pub fn source_id(&self) -> String {
        Self::format_source_id(&self.id)
    }

    pub fn format_source_id(id: &ID) -> String {
        format!("web_crawler:{}", id)
    }
}

#[derive(Validate, GraphQLInputObject)]
pub struct CreateWebCrawlerUrlInput {
    #[validate(url(code = "name", message = "Invalid URL"))]
    pub name: String,
    #[validate(url(code = "url", message = "Invalid URL"))]
    pub url: String,
}

impl NodeType for WebDocument {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "WebCrawlerUrlConnection"
    }

    fn edge_type_name() -> &'static str {
        "WebCrawlerUrlEdge"
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
    ) -> Result<Vec<WebDocument>>;

    async fn create_custom_web_document(&self, name: String, url: String) -> Result<ID>;
    async fn delete_custom_web_document(&self, id: ID) -> Result<()>;
    async fn list_preset_web_documents(&self, active: bool) -> Result<()>;
}
