use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_db::{DbConn, WebCrawlerUrlDAO};
use tabby_schema::{
    job::{JobInfo, JobService},
    web_crawler::{WebCrawlerService, WebCrawlerUrl},
    AsID, AsRowid, Result,
};
use tokio::sync::mpsc::UnboundedSender;

use super::{background_job::BackgroundJobEvent, graphql_pagination_to_filter};

pub fn create(
    db: DbConn,
    job_service: Arc<dyn JobService>,
    sender: UnboundedSender<BackgroundJobEvent>,
) -> impl WebCrawlerService {
    WebCrawlerServiceImpl {
        db,
        sender,
        job_service,
    }
}

struct WebCrawlerServiceImpl {
    db: DbConn,
    job_service: Arc<dyn JobService>,
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

        let mut converted_urls = vec![];

        for url in urls {
            let event = BackgroundJobEvent::WebCrawler(url.url.clone());

            let job_info = self.job_service.get_job_info(event.to_command()).await?;
            converted_urls.push(to_web_crawler_url(url, job_info));
        }
        Ok(converted_urls)
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

fn to_web_crawler_url(value: WebCrawlerUrlDAO, job_info: JobInfo) -> WebCrawlerUrl {
    WebCrawlerUrl {
        id: value.id.as_id(),
        url: value.url.clone(),
        created_at: *value.created_at,
        job_info,
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;
    use tokio::sync::mpsc::UnboundedSender;

    use super::*;
    use crate::background_job::BackgroundJobEvent;

    fn create_fake() -> UnboundedSender<BackgroundJobEvent> {
        tokio::sync::mpsc::unbounded_channel().0
    }

    #[tokio::test]
    async fn test_last_job_run() {
        let db = DbConn::new_in_memory().await.unwrap();
        let background = create_fake();
        let job = Arc::new(crate::service::job::create(db.clone(), background.clone()).await);
        let service = create(db.clone(), job.clone(), background);

        let url = "https://example.com".to_string();
        let id = service.create_web_crawler_url(url.clone()).await.unwrap();

        let command = BackgroundJobEvent::WebCrawler("https://example.com".into()).to_command();

        db.create_job_run("web".into(), command).await.unwrap();

        let urls = service
            .list_web_crawler_urls(None, None, None, None)
            .await
            .unwrap();

        assert_eq!(1, urls.len());
        assert_eq!(id, urls[0].id);
        assert!(urls[0].job_info.last_job_run.is_some());
    }
}
