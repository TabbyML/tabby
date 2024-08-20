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
        let name = doc.get("name").unwrap();
        let url = doc.get("crawlerStart").unwrap();
        preset_web_documents.insert(name.to_string(), url.to_string());
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
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<CustomWebDocument>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let urls = self
            .db
            .list_web_documents(limit, skip_id, backwards, false)
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
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
        include_inactive: bool,
    ) -> Result<Vec<PresetWebDocument>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let urls = self
            .db
            .list_web_documents(limit, skip_id, backwards, true)
            .await?;

        let mut converted_urls = vec![];
        let mut active_urls: HashSet<String> = HashSet::default();

        for url in urls {
            let event = BackgroundJobEvent::WebCrawler(
                CustomWebDocument::format_source_id(&url.id.as_id()),
                url.url.clone(),
            );

            active_urls.insert(url.name.clone());

            let job_info = self.job_service.get_job_info(event.to_command()).await?;
            converted_urls.push(to_preset_web_document(url, job_info));
        }

        if include_inactive {
            for name in self.preset_web_documents.keys() {
                if active_urls.contains(name) {
                    continue;
                }

                converted_urls.push(PresetWebDocument {
                    id: ID::from(name.clone()),
                    job_info: None,
                    updated_at: None,
                });
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
            let id = self
                .db
                .create_web_document(name, url.clone(), true)
                .await?;
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
        id: ID::from(value.name),
        job_info: Some(job_info),
        updated_at: Some(value.updated_at),
    }
}
