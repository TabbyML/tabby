use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_db::{DbConn, WebCrawlerUrlDAO};
use tabby_schema::{
    job::{JobInfo, JobService},
    web_crawler::{WebCrawlerService, WebCrawlerUrl},
    AsID, AsRowid, Result,
};
use crate::service::preset_crawler_urls::get_preset_crawler_urls;
use super::{background_job::BackgroundJobEvent, graphql_pagination_to_filter};

pub async fn create(db: DbConn, job_service: Arc<dyn JobService>) -> impl WebCrawlerService {
    let preset_web_crawler_urls = get_preset_crawler_urls();
    let service = WebCrawlerServiceImpl { db, job_service, preset_web_crawler_urls };
    if let Err(e) = service.init_preset_web_crawler_urls().await {
        panic!("failed to init preset web crawler urls because of {:?}", e);
    }
    service
}

struct WebCrawlerServiceImpl {
    db: DbConn,
    job_service: Arc<dyn JobService>,
    preset_web_crawler_urls: BTreeMap<String, String>,
}

impl WebCrawlerServiceImpl {
    pub async fn init_preset_web_crawler_urls(&self) -> Result<()> {
        let origin_urls = self.db.list_web_crawler_urls(None, None, false, None, Some(true)).await?;
        let mut origin_urls_index: HashSet<String> = HashSet::default();
        for url in origin_urls {
            origin_urls_index.insert(url.url);
        }
        for (name, url) in &self.preset_web_crawler_urls {
            if origin_urls_index.contains(url) {
                continue;
            }

            self.db.create_web_crawler_url(name.clone(), url.clone(), false, true).await?;
        }
        Ok(())
    }
}

#[async_trait]
impl WebCrawlerService for WebCrawlerServiceImpl {
    async fn list_web_crawler_urls(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
        active: Option<bool>,
        preset: Option<bool>,
    ) -> Result<Vec<WebCrawlerUrl>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let urls = self
            .db
            .list_web_crawler_urls(limit, skip_id, backwards, active, preset)
            .await?;

        let mut converted_urls = vec![];

        for url in urls {
            let event = BackgroundJobEvent::WebCrawler(
                WebCrawlerUrl::format_source_id(&url.id.as_id()),
                url.url.clone(),
            );

            let job_info = self.job_service.get_job_info(event.to_command()).await?;
            converted_urls.push(to_web_crawler_url(url, job_info));
        }
        Ok(converted_urls)
    }

    async fn create_web_crawler_url(&self, url: String) -> Result<ID> {
        let id = self.db.create_web_crawler_url("".to_string(), url.clone(), true, false).await?;

        let _ = self
            .job_service
            .trigger(
                BackgroundJobEvent::WebCrawler(WebCrawlerUrl::format_source_id(&id.as_id()), url)
                    .to_command(),
            )
            .await;

        Ok(id.as_id())
    }

    async fn delete_web_crawler_url(&self, id: ID) -> Result<()> {
        self.db.delete_web_crawler_url(id.as_rowid()?).await?;
        self.job_service
            .trigger(BackgroundJobEvent::IndexGarbageCollection.to_command())
            .await?;
        Ok(())
    }
    async fn update_active_web_crawler_url(&self,  id: ID, active: bool) -> Result<()> {
        let url = self.db.update_active_web_crawler_url(id.as_rowid()?, active).await?;
        if active {
            let _ = self
                .job_service
                .trigger(
                    BackgroundJobEvent::WebCrawler(WebCrawlerUrl::format_source_id(&id), url)
                        .to_command(),
                )
                .await;
        } else {
            self.job_service
                .trigger(BackgroundJobEvent::IndexGarbageCollection.to_command())
                .await?;
        }
        Ok(())
    }
}

fn to_web_crawler_url(value: WebCrawlerUrlDAO, job_info: JobInfo) -> WebCrawlerUrl {
    WebCrawlerUrl {
        id: value.id.as_id(),
        name: value.web_name,
        is_preset: value.is_preset,
        url: value.url.clone(),
        created_at: value.created_at,
        job_info,
        is_active: value.active,
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;

    use super::*;
    use crate::background_job::BackgroundJobEvent;

    #[tokio::test]
    async fn test_last_job_run() {
        let db = DbConn::new_in_memory().await.unwrap();
        let job = Arc::new(crate::service::job::create(db.clone()).await);
        let service = create(db.clone(), job.clone()).await;

        let url = "https://example.com".to_string();
        let id = service.create_web_crawler_url(url.clone()).await.unwrap();

        let command =
            BackgroundJobEvent::WebCrawler("id".into(), "https://example.com".into()).to_command();

        db.create_job_run("web".into(), command).await.unwrap();

        let urls = service
            .list_web_crawler_urls(None, None, None, None, Some(true), None)
            .await
            .unwrap();

        assert_eq!(1, urls.len());
        assert_eq!(id, urls[0].id);
        assert!(urls[0].job_info.last_job_run.is_some());
    }
}
