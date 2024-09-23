use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{anyhow, Context};
use async_trait::async_trait;
use juniper::ID;
use tabby_db::{DbConn};
use tabby_schema::{
    job::{JobInfo, JobService}, notion_documents::{CreateNotionDocumentInput, NotionDocument, NotionDocumentService},  AsID, AsRowid, CoreError, Result
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
    

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tabby_db::DbConn;
    use tabby_schema::notion_documents::NotionDocumentService;
    use super::{create,CreateNotionDocumentInput};
    use tabby_db::notion_documents:: NotionDocumentType;

    #[tokio::test]
    async fn test_list_notion_documents() {
        let db = DbConn::new_in_memory().await.unwrap();
        let job = Arc::new(crate::service::job::create(db.clone()).await);
        let service = create(db.clone(), job.clone());

        let id = service
            .create_notion_document(
                CreateNotionDocumentInput{
                    name: "example".to_string(),
                    integration_id: "example".to_string(),
                    integration_type: NotionDocumentType::Database,
                    access_token: "example".to_string(),
                }
            )
            .await
            .unwrap();

       
        let docs = service
            .list_notion_documents(None, None, None, None, None)
            .await;
        if docs.is_err (){
            panic!("{}", docs.err().unwrap());
        }
        let docs = docs.unwrap();

        assert_eq!(1, docs.len());
        assert_eq!(id, docs[0].id);
        assert!(docs[0].job_info.last_job_run.is_some());

        service.delete_notion_document(id).await.unwrap();
        let urls = service
            .list_notion_documents(None, None, None, None, None)
            .await
            .unwrap();

        assert_eq!(0, urls.len());

        let _id1 = service
            .create_notion_document(
                CreateNotionDocumentInput{
                    name: "example1".to_string(),
                    integration_id: "example1".to_string(),
                    integration_type: NotionDocumentType::Database,
                    access_token: "example1".to_string(),
                }
            )
            .await
            .unwrap();
        let id2 = service
            .create_notion_document(
                CreateNotionDocumentInput{
                    name: "example2".to_string(),
                    integration_id: "example2".to_string(),
                    integration_type: NotionDocumentType::Database,
                    access_token: "example2".to_string(),
                }
            )
            .await
            .unwrap();
        let id3 = service
            .create_notion_document(
                CreateNotionDocumentInput{
                    name: "example3".to_string(),
                    integration_id: "example3".to_string(),
                    integration_type: NotionDocumentType::Database,
                    access_token: "example3".to_string(),
                }
            )
            .await
            .unwrap();
        let docs = service
            .list_notion_documents(Some(vec![id2.clone(), id3.clone()]), None, None, None, None)
            .await
            .unwrap();
        assert_eq!(2, docs.len());

        service.delete_notion_document(_id1).await.unwrap();
        service.delete_notion_document(id2.clone()).await.unwrap();
        service.delete_notion_document(id3.clone()).await.unwrap();
        let docs = service
            .list_notion_documents(Some(vec![id2.clone(), id3.clone()]), None, None, None, None)
            .await
            .unwrap();
        assert_eq!(0, docs.len());

    }

}
