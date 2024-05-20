use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    integration::{Integration, IntegrationKind, IntegrationService},
    AsID, AsRowid, DbEnum, Result,
};
use tokio::sync::mpsc::UnboundedSender;

use super::graphql_pagination_to_filter;
use crate::service::background_job::BackgroundJobEvent;

struct IntegrationServiceImpl {
    db: DbConn,
    background_job: UnboundedSender<BackgroundJobEvent>,
}

pub fn create(
    db: DbConn,
    background_job: UnboundedSender<BackgroundJobEvent>,
) -> impl IntegrationService {
    IntegrationServiceImpl { db, background_job }
}

#[async_trait]
impl IntegrationService for IntegrationServiceImpl {
    async fn create_integration(
        &self,
        kind: IntegrationKind,
        display_name: String,
        access_token: String,
        url_base: Option<String>,
    ) -> Result<ID> {
        let id = self
            .db
            .create_integration(
                kind.as_enum_str().to_string(),
                display_name,
                access_token,
                url_base.unwrap_or_else(|| kind.default_url_base().to_string()),
            )
            .await?;
        let id = id.as_id();
        let _ = self
            .background_job
            .send(BackgroundJobEvent::SyncThirdPartyRepositories(id.clone()));
        Ok(id)
    }

    async fn delete_integration(&self, id: ID, kind: IntegrationKind) -> Result<()> {
        self.db
            .delete_integration(id.as_rowid()?, kind.as_enum_str())
            .await?;
        Ok(())
    }

    async fn update_integration(
        &self,
        id: ID,
        kind: IntegrationKind,
        display_name: String,
        access_token: Option<String>,
    ) -> Result<()> {
        self.db
            .update_integration(
                id.as_rowid()?,
                kind.as_enum_str(),
                display_name,
                access_token,
            )
            .await?;
        Ok(())
    }

    async fn list_integrations(
        &self,
        ids: Option<Vec<ID>>,
        kind: Option<IntegrationKind>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Integration>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let ids = ids
            .unwrap_or_default()
            .into_iter()
            .map(|id| id.as_rowid())
            .collect::<Result<_, _>>()?;
        let kind = kind.map(|kind| kind.as_enum_str().to_string());
        let integrations = self
            .db
            .list_integrations(ids, kind, limit, skip_id, backwards)
            .await?;

        Ok(integrations
            .into_iter()
            .map(Integration::try_from)
            .collect::<Result<_, _>>()?)
    }

    async fn get_integration(&self, id: ID) -> Result<Integration> {
        Ok(self.db.get_integration(id.as_rowid()?).await?.try_into()?)
    }

    async fn update_integration_sync_status(&self, id: ID, error: Option<String>) -> Result<()> {
        self.db
            .update_integration_error(id.as_rowid()?, error)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use tabby_schema::integration::IntegrationStatus;

    use super::*;

    fn create_fake() -> UnboundedSender<BackgroundJobEvent> {
        let (sender, _) = tokio::sync::mpsc::unbounded_channel();
        sender
    }

    #[tokio::test]
    async fn test_integration_crud() {
        let background = create_fake();
        let db = DbConn::new_in_memory().await.unwrap();
        let integration = Arc::new(create(db, background));

        let id = integration
            .create_integration(IntegrationKind::Gitlab, "id".into(), "secret".into(), None)
            .await
            .unwrap();

        // Test listing gitlab providers
        let providers = integration
            .list_integrations(None, Some(IntegrationKind::Gitlab), None, None, None, None)
            .await
            .unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].access_token, "secret");

        // Test updating gitlab provider
        integration
            .update_integration(id.clone(), IntegrationKind::Gitlab, "id2".into(), None)
            .await
            .unwrap();

        let provider = integration.get_integration(id.clone()).await.unwrap();
        assert_eq!(provider.display_name, "id2");
        assert_eq!(provider.status, IntegrationStatus::Pending);

        // Test updating error status for gitlab provider
        tokio::time::sleep(Duration::from_secs(1)).await;
        integration
            .update_integration_sync_status(id.clone(), Some("error".into()))
            .await
            .unwrap();

        let provider = integration.get_integration(id.clone()).await.unwrap();
        assert_eq!(provider.status, IntegrationStatus::Failed);

        // Test successful status (no error)
        integration
            .update_integration_sync_status(id.clone(), None)
            .await
            .unwrap();

        let provider = integration.get_integration(id.clone()).await.unwrap();
        assert_eq!(provider.status, IntegrationStatus::Ready);

        // Deleting using github integration kind should fail since this is a gitlab integration
        assert!(integration
            .delete_integration(id.clone(), IntegrationKind::Github)
            .await
            .is_err());

        // Test successful deletion
        integration
            .delete_integration(id.clone(), IntegrationKind::Gitlab)
            .await
            .unwrap();

        assert_eq!(
            0,
            integration
                .list_integrations(None, None, None, None, None, None)
                .await
                .unwrap()
                .len()
        );
    }
}
