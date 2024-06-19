use async_trait::async_trait;
use juniper::ID;
use tabby_db::{DbConn, WebCrawlerUrlDAO};
use tabby_schema::{
    job::JobInfo,
    web_crawler::{WebCrawlerService, WebCrawlerUrl},
    AsID, AsRowid, Result,
};
use tokio::sync::mpsc::UnboundedSender;

use super::{background_job::BackgroundJobEvent, graphql_pagination_to_filter};

pub fn create(db: DbConn, sender: UnboundedSender<BackgroundJobEvent>) -> impl WebCrawlerService {
    WebCrawlerServiceImpl { db, sender }
}

struct WebCrawlerServiceImpl {
    db: DbConn,
    sender: UnboundedSender<BackgroundJobEvent>,
}

#[async_trait]
impl WebCrawlerService for WebCrawlerServiceImpl {
    async fn list_web_crawler_urls(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<WebCrawlerUrl>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let urls = self
            .db
            .list_web_crawler_urls(limit, skip_id, backwards)
            .await?;
        Ok(urls.into_iter().map(to_web_crawler_url).collect())
    }

    async fn create_web_crawler_url(&self, url: String) -> Result<ID> {
        let id = self.db.create_web_crawler_url(url.clone()).await?;

        let _ = self.sender.send(BackgroundJobEvent::WebCrawler(url));

        Ok(id.as_id())
    }

    async fn delete_web_crawler_url(&self, id: ID) -> Result<()> {
        self.db.delete_web_crawler_url(id.as_rowid()?).await?;
        Ok(())
    }
}

fn to_web_crawler_url(value: WebCrawlerUrlDAO) -> WebCrawlerUrl {
    WebCrawlerUrl {
        id: value.id.as_id(),
        url: value.url.clone(),
        created_at: *value.created_at,
        job_info: JobInfo {
            last_job_run: None,
            command: serde_json::to_string(&BackgroundJobEvent::WebCrawler(value.url))
                .expect("Failed to serialize job command"),
        },
    }
}
