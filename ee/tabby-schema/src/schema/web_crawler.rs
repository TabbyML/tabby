use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLInputObject, GraphQLObject, ID};
use validator::Validate;

use crate::{job::JobInfo, juniper::relay::NodeType, Context, Result};

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct WebCrawlerUrl {
    pub url: String,
    pub id: ID,
    pub created_at: DateTime<Utc>,
    pub job_info: JobInfo,
}

impl WebCrawlerUrl {
    pub fn source_id(&self) -> String {
        Self::format_source_id(&self.id)
    }

    pub fn format_source_id(id: &ID) -> String {
        format!("web_crawler:{}", id)
    }
}

#[derive(Validate, GraphQLInputObject)]
pub struct CreateWebCrawlerUrlInput {
    #[validate(url(code = "url", message = "Invalid URL"))]
    pub url: String,
}

impl NodeType for WebCrawlerUrl {
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
pub trait WebCrawlerService: Send + Sync {
    async fn list_web_crawler_urls(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<WebCrawlerUrl>>;

    async fn create_web_crawler_url(&self, url: String) -> Result<ID>;
    async fn delete_web_crawler_url(&self, id: ID) -> Result<()>;
}
