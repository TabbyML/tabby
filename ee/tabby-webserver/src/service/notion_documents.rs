use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{anyhow, Context};
use async_trait::async_trait;
use juniper::ID;
use tabby_db::{DbConn};
use tabby_schema::{
    job::{JobInfo, JobService}, notion_documents::{self, CreateNotionDocumentInput, NotionDocument, NotionDocumentService},  AsID, AsRowid, CoreError, Result
};

use super::{
    background_job::{BackgroundJobEvent, NotionJob},
    graphql_pagination_to_filter,
    preset_web_documents_data::PRESET_WEB_DOCUMENTS_DATA,
};

pub fn create(db: DbConn, job_service: Arc<dyn JobService>) -> impl NotionDocumentService {
    NotionServiceImpl {
                db,
                job_service,
            }
}


struct NotionServiceImpl {
    db: DbConn,
    job_service: Arc<dyn JobService>,
}

#[async_trait]
impl NotionDocumentService for NotionServiceImpl {
    async fn list_notion_documents(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<NotionDocument>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let ids = ids.map(|x| {
            x.iter()
                .filter_map(|x| x.as_rowid().ok())
                .collect::<Vec<_>>()
        });
        let notion_documents_dao = self
            .db
            .list_notion_documents(ids, limit, skip_id, backwards)
            .await?;

        let mut notion_documents= vec![];

        for doc in notion_documents_dao {
            let event: BackgroundJobEvent = BackgroundJobEvent::NotionIntregration(NotionJob{
                source_id: NotionDocument::format_source_id(&doc.id.as_id()),
                integration_id: doc.integration_id.clone(),
                integration_type: doc.integration_type.clone(),
                access_token: doc.access_token.clone(),
            });

            let job_info = self.job_service.get_job_info(event.to_command()).await?;
            notion_documents.push(NotionDocument{
                id: doc.id.as_id(),
                name: doc.name,
                integration_id: doc.integration_id.clone(),
                integration_type: doc.integration_type.clone(),
                access_token: doc.access_token.clone(),
                created_at: doc.created_at,
                updated_at: doc.updated_at,
                job_info,
            });
            };
        Ok(notion_documents)
    }

    async fn create_notion_document(&self, input: CreateNotionDocumentInput) -> Result<ID> {
       
        let id = self
            .db
            .create_notion_document(input.name.clone(), input.integration_id.clone(), input.integration_type.clone(), input.access_token.clone())
            .await?;
        let _ = self
            .job_service
            .trigger(
                BackgroundJobEvent::NotionIntregration(NotionJob{
                    source_id: NotionDocument::format_source_id(&id.as_id()),
                    integration_id: input.integration_id.clone(),
                    integration_type: input.integration_type.clone(),
                    access_token: input.access_token.clone(),
                })
                .to_command(),
            )
            .await;
        Ok(id.as_id())
    }

    async fn delete_notion_document(&self, id: ID) -> Result<bool> {
        let rowid = id.as_rowid()?;
        let doc = {
            let mut x = self
                .db
                .list_notion_documents(Some(vec![rowid]), None, None, false)
                .await?;
            x.pop().context("notion document doesn't exist")?
        };
        self.db.delete_notion_document(rowid).await?;
        self.job_service
            .clear(
                BackgroundJobEvent::NotionIntregration(NotionJob{
                    source_id: NotionDocument::format_source_id(&doc.id.as_id()),
                    integration_id: doc.integration_id,
                    integration_type: doc.integration_type,
                    access_token: doc.access_token,
                })
                .to_command(),
            )
            .await?;
        self.job_service
            .trigger(BackgroundJobEvent::IndexGarbageCollection.to_command())
            .await?;
        Ok(true)
    }
}
    
