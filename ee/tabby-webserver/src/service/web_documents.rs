use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::anyhow;
use async_trait::async_trait;
use juniper::ID;
use tabby_db::{DbConn, WebDocumentDAO};
use tabby_schema::{
    job::{JobInfo, JobService},
    web_documents::{CustomWebDocument, PresetWebDocument, WebDocumentService},
    AsID, AsRowid, CoreError, Result,
};

use super::{
    background_job::BackgroundJobEvent, graphql_pagination_to_filter,
    preset_web_documents_data::PRESET_WEB_DOCUMENTS_DATA,
};

pub fn create(db: DbConn, job_service: Arc<dyn JobService>) -> impl WebDocumentService {
    let data: serde_json::Value = serde_json::from_str(PRESET_WEB_DOCUMENTS_DATA).unwrap();
    let mut preset_web_documents = HashMap::new();
    for doc in data.as_array().unwrap() {
        let name = doc.get("name").unwrap().to_string();
        let url = doc.get("crawlerStart").unwrap().to_string();
        preset_web_documents.insert(
            String::from(&name[1..name.len() - 1]),
            String::from(&url[1..url.len() - 1]),
        );
    }
    WebDocumentServiceImpl {
        db,
        job_service,
        preset_web_documents,
    }
}

struct WebDocumentServiceImpl {
    db: DbConn,
    job_service: Arc<dyn JobService>,
    preset_web_documents: HashMap<String, String>,
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
            let event = BackgroundJobEvent::WebCrawler(
                CustomWebDocument::format_source_id(&url.id.as_id()),
                url.url.clone(),
            );

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
                BackgroundJobEvent::WebCrawler(
                    CustomWebDocument::format_source_id(&id.as_id()),
                    url,
                )
                .to_command(),
            )
            .await;
        Ok(id.as_id())
    }

    async fn delete_custom_web_document(&self, id: ID) -> Result<()> {
        self.db.delete_web_document(id.as_rowid()?).await?;
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
        is_active: bool,
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

        for url in urls {
            active_urls.insert(url.name.clone());

            if is_active {
                let event = BackgroundJobEvent::WebCrawler(
                    CustomWebDocument::format_source_id(&url.id.as_id()),
                    url.url.clone(),
                );

                let job_info = self.job_service.get_job_info(event.to_command()).await?;
                converted_urls.push(to_preset_web_document(url, job_info));
            }
        }

        if !is_active {
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

    async fn set_preset_web_documents_active(&self, name: String, active: bool) -> Result<()> {
        if !active {
            self.db.deactivate_preset_web_document(name).await?;
            self.job_service
                .trigger(BackgroundJobEvent::IndexGarbageCollection.to_command())
                .await?;
            Ok(())
        } else {
            let Some(url) = self.preset_web_documents.get(&name) else {
                return Err(CoreError::Other(anyhow!(format!(
                    "name {} does not exist",
                    name
                ))));
            };
            let id = self.db.create_web_document(name, url.clone(), true).await?;
            let _ = self
                .job_service
                .trigger(
                    BackgroundJobEvent::WebCrawler(
                        CustomWebDocument::format_source_id(&id.as_id()),
                        url.clone(),
                    )
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

        let command = BackgroundJobEvent::WebCrawler(id.to_string(), "https://example.com".into())
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
            .set_preset_web_documents_active("React".to_string(), true)
            .await
            .unwrap();

        let command =
            BackgroundJobEvent::WebCrawler("id".into(), "https://react.dev/reference/".into())
                .to_command();

        db.create_job_run("preset".into(), command).await.unwrap();
        service
            .set_preset_web_documents_active("Qwik".to_string(), true)
            .await
            .unwrap();
        let urls = service
            .list_preset_web_documents(None, None, None, None, None, true)
            .await
            .unwrap();
        assert_eq!(2, urls.len());

        let urls = service
            .list_preset_web_documents(None, None, None, None, None, false)
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
                true,
            )
            .await
            .unwrap();
        assert_eq!(1, urls.len());

        service
            .set_preset_web_documents_active("React".to_string(), false)
            .await
            .unwrap();
        service
            .set_preset_web_documents_active("Qwik".to_string(), false)
            .await
            .unwrap();
        let urls = service
            .list_preset_web_documents(None, None, None, None, None, false)
            .await
            .unwrap();
        assert_eq!(384, urls.len());

        service.delete_custom_web_document(id).await.unwrap();
        let urls = service
            .list_custom_web_documents(None, None, None, None, None)
            .await
            .unwrap();

        assert_eq!(0, urls.len());

        let id1 = service
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
                false,
            )
            .await
            .unwrap();
        assert_eq!(2, urls.len());
    }
}
