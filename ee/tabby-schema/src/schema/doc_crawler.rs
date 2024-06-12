use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLInputObject, GraphQLObject, ID};
use validator::Validate;

use crate::{juniper::relay::NodeType, Context, Result};

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct DocCrawlerUrl {
    pub url: String,
    pub id: ID,
    pub created_at: DateTime<Utc>,
}

#[derive(Validate, GraphQLInputObject)]
pub struct CreateDocCrawlerUrlInput {
    #[validate(url)]
    pub url: String,
}

impl NodeType for DocCrawlerUrl {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "DocCrawlerUrlConnection"
    }

    fn edge_type_name() -> &'static str {
        "DocCrawlerUrlEdge"
    }
}

#[async_trait]
pub trait DocCrawlerService: Send + Sync {
    async fn list_doc_crawler_urls(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<DocCrawlerUrl>>;

    async fn create_doc_crawler_url(&self, url: String) -> Result<ID>;
    async fn delete_doc_crawler_url(&self, id: ID) -> Result<()>;
}
