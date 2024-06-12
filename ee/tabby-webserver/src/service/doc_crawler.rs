use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    doc_crawler::{DocCrawlerService, DocCrawlerUrl},
    AsID, AsRowid, Result,
};

use super::graphql_pagination_to_filter;

pub fn create(db: DbConn) -> impl DocCrawlerService {
    DocCrawlerServiceImpl { db }
}

struct DocCrawlerServiceImpl {
    db: DbConn,
}

#[async_trait]
impl DocCrawlerService for DocCrawlerServiceImpl {
    async fn list_doc_crawler_urls(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<DocCrawlerUrl>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let urls = self
            .db
            .list_doc_crawler_urls(limit, skip_id, backwards)
            .await?;
        Ok(urls.into_iter().map(DocCrawlerUrl::from).collect())
    }

    async fn create_doc_crawler_url(&self, url: String) -> Result<ID> {
        let id = self.db.create_doc_crawler_url(url).await?;
        Ok(id.as_id())
    }

    async fn delete_doc_crawler_url(&self, id: ID) -> Result<()> {
        self.db.delete_doc_crawler_url(id.as_rowid()?).await?;
        Ok(())
    }
}
