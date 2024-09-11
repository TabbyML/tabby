use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{anyhow, Context};
use async_trait::async_trait;
use juniper::ID;
use tabby_db::{DbConn, WebDocumentDAO};
use tabby_schema::{
    job::{JobInfo, JobService},
    web_documents::{CustomWebDocument, PresetWebDocument, WebDocumentService},
    AsID, AsRowid, CoreError, Result,
};

use super::{
    background_job::{BackgroundJobEvent, WebCrawlerJob},
    graphql_pagination_to_filter,
    preset_web_documents_data::PRESET_WEB_DOCUMENTS_DATA,
};

pub fn create(db: DbConn, job_service: Arc<dyn JobService>) -> impl WebDocumentService {
    create_impl(db, job_service)
}

fn read_preset_document_urls(json_str: &str) -> HashMap<String, (String, String)> {
    let data: serde_json::Value = serde_json::from_str(json_str).unwrap();
    let mut preset_web_documents = HashMap::new();
    for doc in data.as_array().unwrap() {
        let name = doc.get("name").unwrap().as_str().unwrap().to_owned();
        let url = doc
            .get("crawlerStart")
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        let url_prefix = doc
            .get("crawlerPrefix")
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        preset_web_documents.insert(name, (url, url_prefix));
    }

    preset_web_documents
}

fn create_impl(db: DbConn, job_service: Arc<dyn JobService>) -> WebDocumentServiceImpl {
    WebDocumentServiceImpl {
        db,
        job_service,
        preset_web_documents: read_preset_document_urls(PRESET_WEB_DOCUMENTS_DATA),
    }
}

struct WebDocumentServiceImpl {
    db: DbConn,
    job_service: Arc<dyn JobService>,
    preset_web_documents: HashMap<String, (String, String)>,
}

#[async_trait]
impl WebDocumentService for WebDocumentServiceImpl {
    async fn list_custom_web_documents(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<CustomWebDocument>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let ids = ids.map(|x| {
            x.iter()
                .filter_map(|x| x.as_rowid().ok())
                .collect::<Vec<_>>()
        });
        let urls = self
            .db
            .list_custom_web_documents(ids, limit, skip_id, backwards)
            .await?;

        let mut converted_urls = vec![];

        for url in urls {
            let event = BackgroundJobEvent::WebCrawler(WebCrawlerJob::new(
                CustomWebDocument::format_source_id(&url.id.as_id()),
                url.url.clone(),
                None,
            ));

            let job_info = self.job_service.get_job_info(event.to_command()).await?;
            converted_urls.push(to_custom_web_document(url, job_info));
        }
        Ok(converted_urls)
    }

    async fn create_custom_web_document(&self, name: String, url: String) -> Result<ID> {
        if self.preset_web_documents.contains_key(&name) {
            return Err(CoreError::Other(anyhow!(format!(
                "name: {} is conflicts with preset document",
                name
            ))));
        }
        let id = self
            .db
            .create_web_document(name, url.clone(), false)
            .await?;
        let _ = self
            .job_service
            .trigger(
                BackgroundJobEvent::WebCrawler(WebCrawlerJob::new(
                    CustomWebDocument::format_source_id(&id.as_id()),
                    url,
                    None,
                ))
                .to_command(),
            )
            .await;
        Ok(id.as_id())
    }

    async fn delete_custom_web_document(&self, id: ID) -> Result<()> {
        let rowid = id.as_rowid()?;
        let webdoc = {
            let mut x = self
                .db
                .list_custom_web_documents(Some(vec![rowid]), None, None, false)
                .await?;

            x.pop().context("web document doesn't exist")?
        };
        self.db.delete_web_document(rowid).await?;
        self.job_service
            .clear(
                BackgroundJobEvent::WebCrawler(WebCrawlerJob::new(
                    CustomWebDocument::format_source_id(&webdoc.id.as_id()),
                    webdoc.url,
                    None,
                ))
                .to_command(),
            )
            .await?;
        self.job_service
            .trigger(BackgroundJobEvent::IndexGarbageCollection.to_command())
            .await?;
        Ok(())
    }

    async fn list_preset_web_documents(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
        is_active: Option<bool>,
    ) -> Result<Vec<PresetWebDocument>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let names: Option<Vec<String>> =
            ids.map(|ids| ids.into_iter().map(|id| id.to_string()).collect::<Vec<_>>());
        let urls = self
            .db
            .list_preset_web_documents(names.clone(), limit, skip_id, backwards)
            .await?;

        let mut converted_urls = vec![];
        let mut active_urls: HashSet<String> = HashSet::default();
        let include_active = is_active.unwrap_or(true);
        let include_inactive = !is_active.unwrap_or(false);

        for url in urls {
            active_urls.insert(url.name.clone());

            if include_active {
                let Some((_, url_prefix)) = self.preset_web_documents.get(&url.name) else {
                    return Err(CoreError::Other(anyhow!(format!(
                        "name {} does not exist",
                        url.name
                    ))));
                };

                let event = BackgroundJobEvent::WebCrawler(WebCrawlerJob::new(
                    PresetWebDocument::format_source_id(&url.name),
                    url.url.clone(),
                    Some(url_prefix.clone()),
                ));

                let job_info = self.job_service.get_job_info(event.to_command()).await?;
                converted_urls.push(to_preset_web_document(url, job_info));
            }
        }

        if include_inactive {
            if let Some(names) = names {
                for name in names {
                    if active_urls.contains(&name) || !self.preset_web_documents.contains_key(&name)
                    {
                        continue;
                    }
                    converted_urls.push(PresetWebDocument {
                        id: ID::from(name.clone()),
                        name,
                        job_info: None,
                        updated_at: None,
                        is_active: false,
                    });
                }
            } else {
                for name in self.preset_web_documents.keys() {
                    if active_urls.contains(name) {
                        continue;
                    }
                    converted_urls.push(PresetWebDocument {
                        id: ID::from(name.clone()),
                        name: name.clone(),
                        job_info: None,
                        updated_at: None,
                        is_active: false,
                    });
                }
            }
        }

        Ok(converted_urls)
    }

    async fn set_preset_web_documents_active(&self, id: ID, active: bool) -> Result<()> {
        let name = id.to_string();
        let Some((url, url_prefix)) = self.preset_web_documents.get(&name) else {
            return Err(CoreError::Other(anyhow!(format!(
                "name {} does not exist",
                name
            ))));
        };

        if !active {
            self.db.deactivate_preset_web_document(&name).await?;
            self.job_service
                .clear(
                    BackgroundJobEvent::WebCrawler(WebCrawlerJob::new(
                        PresetWebDocument::format_source_id(&name),
                        url.clone(),
                        Some(url_prefix.clone()),
                    ))
                    .to_command(),
                )
                .await?;
            self.job_service
                .trigger(BackgroundJobEvent::IndexGarbageCollection.to_command())
                .await?;
            Ok(())
        } else {
            let _id = self
                .db
                .create_web_document(name.clone(), url.clone(), true)
                .await?;
            let _ = self
                .job_service
                .trigger(
                    BackgroundJobEvent::WebCrawler(WebCrawlerJob::new(
                        PresetWebDocument::format_source_id(&name),
                        url.clone(),
                        Some(url_prefix.clone()),
                    ))
                    .to_command(),
                )
                .await;
            Ok(())
        }
    }
}

fn to_custom_web_document(value: WebDocumentDAO, job_info: JobInfo) -> CustomWebDocument {
    CustomWebDocument {
        id: value.id.as_id(),
        url: value.url.clone(),
        created_at: value.created_at,
        updated_at: value.updated_at,
        job_info,
        name: value.name,
    }
}

fn to_preset_web_document(value: WebDocumentDAO, job_info: JobInfo) -> PresetWebDocument {
    PresetWebDocument {
        id: ID::from(value.name.clone()),
        name: value.name,
        job_info: Some(job_info),
        updated_at: Some(value.updated_at),
        is_active: true,
    }
}

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;
    use tabby_schema::constants::WEB_DOCUMENT_NAME_REGEX;

    use super::*;
    use crate::background_job::BackgroundJobEvent;

    #[tokio::test]
    async fn test_list_web_documents() {
        let db = DbConn::new_in_memory().await.unwrap();
        let job = Arc::new(crate::service::job::create(db.clone()).await);
        let service = create(db.clone(), job.clone());

        let url = "https://example.com".to_string();
        let id = service
            .create_custom_web_document("example".to_string(), url.clone())
            .await
            .unwrap();

        let command = BackgroundJobEvent::WebCrawler(WebCrawlerJob::new(
            id.to_string(),
            "https://example.com".into(),
            None,
        ))
        .to_command();

        db.create_job_run("web".into(), command).await.unwrap();

        let urls = service
            .list_custom_web_documents(None, None, None, None, None)
            .await
            .unwrap();

        assert_eq!(1, urls.len());
        assert_eq!(id, urls[0].id);
        assert!(urls[0].job_info.last_job_run.is_some());

        service
            .set_preset_web_documents_active(ID::from("React".to_string()), true)
            .await
            .unwrap();

        let command = BackgroundJobEvent::WebCrawler(WebCrawlerJob::new(
            "id".into(),
            "https://react.dev/reference/".into(),
            None,
        ))
        .to_command();

        db.create_job_run("preset".into(), command).await.unwrap();
        service
            .set_preset_web_documents_active(ID::from("Qwik".to_string()), true)
            .await
            .unwrap();
        let urls = service
            .list_preset_web_documents(None, None, None, None, None, Some(true))
            .await
            .unwrap();
        assert_eq!(2, urls.len());

        let urls = service
            .list_preset_web_documents(None, None, None, None, None, Some(false))
            .await
            .unwrap();
        assert_eq!(382, urls.len());

        let urls = service
            .list_preset_web_documents(
                Some(vec![ID::from("React".to_string())]),
                None,
                None,
                None,
                None,
                Some(true),
            )
            .await
            .unwrap();
        assert_eq!(1, urls.len());
        assert!(urls[0].is_active);

        let urls = service
            .list_preset_web_documents(None, None, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(384, urls.len());

        service
            .set_preset_web_documents_active(ID::from("React".to_string()), false)
            .await
            .unwrap();
        service
            .set_preset_web_documents_active(ID::from("Qwik".to_string()), false)
            .await
            .unwrap();
        let urls = service
            .list_preset_web_documents(None, None, None, None, None, Some(false))
            .await
            .unwrap();
        assert_eq!(384, urls.len());

        service.delete_custom_web_document(id).await.unwrap();
        let urls = service
            .list_custom_web_documents(None, None, None, None, None)
            .await
            .unwrap();

        assert_eq!(0, urls.len());

        let _id1 = service
            .create_custom_web_document("example1".to_string(), "https://example1.com".to_string())
            .await
            .unwrap();
        let id2 = service
            .create_custom_web_document("example2".to_string(), "https://example2.com".to_string())
            .await
            .unwrap();
        let id3 = service
            .create_custom_web_document("example3".to_string(), "https://example3.com".to_string())
            .await
            .unwrap();

        let urls = service
            .list_custom_web_documents(Some(vec![id2, id3]), None, None, None, None)
            .await
            .unwrap();
        assert_eq!(2, urls.len());

        let urls = service
            .list_preset_web_documents(
                Some(vec![
                    ID::from("React".to_string()),
                    ID::from("Qwik".to_string()),
                ]),
                None,
                None,
                None,
                None,
                Some(false),
            )
            .await
            .unwrap();
        assert_eq!(2, urls.len());
    }

    #[tokio::test]
    async fn test_preset_web_documents_validate() {
        let db = DbConn::new_in_memory().await.unwrap();
        let job = Arc::new(crate::service::job::create(db.clone()).await);
        let service = create_impl(db.clone(), job.clone());
        for name in service.preset_web_documents.keys() {
            assert!(
                WEB_DOCUMENT_NAME_REGEX.is_match(name.as_str()),
                "name: {}",
                name
            );
        }
    }

    #[test]
    fn test_read_preset_document_urls() {
        const PRESET_WEB_DOCUMENTS_DATA: &str = r#"[
        { "name": "Serp Google Scholar API", "crawlerStart": "https://serpapi.com/google-scholar-api", "crawlerPrefix": "https://serpapi.com/google-scholar-api" },
        { "name": "Novu", "crawlerStart": "https://docs.novu.co/", "crawlerPrefix": "https://docs.novu.co/" }
    ]"#;

        let expected_output: HashMap<String, (String, String)> = [
            (
                "Serp Google Scholar API".to_string(),
                (
                    "https://serpapi.com/google-scholar-api".to_string(),
                    "https://serpapi.com/google-scholar-api".to_string(),
                ),
            ),
            (
                "Novu".to_string(),
                (
                    "https://docs.novu.co/".to_string(),
                    "https://docs.novu.co/".to_string(),
                ),
            ),
        ]
        .iter()
        .cloned()
        .collect();

        let result = read_preset_document_urls(PRESET_WEB_DOCUMENTS_DATA);
        assert_eq!(result, expected_output);
    }
}
