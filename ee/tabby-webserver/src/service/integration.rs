use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    integration::{Integration, IntegrationKind, IntegrationService},
    job::JobService,
    AsID, AsRowid, DbEnum, Result,
};

use super::graphql_pagination_to_filter;
use crate::{bail, service::background_job::BackgroundJobEvent};

struct IntegrationServiceImpl {
    db: DbConn,
    job: Arc<dyn JobService>,
}

pub fn create(db: DbConn, job: Arc<dyn JobService>) -> impl IntegrationService {
    IntegrationServiceImpl { db, job }
}

#[async_trait]
impl IntegrationService for IntegrationServiceImpl {
    async fn create_integration(
        &self,
        kind: IntegrationKind,
        display_name: String,
        access_token: String,
        api_base: Option<String>,
    ) -> Result<ID> {
        if kind.is_self_hosted() && api_base.is_none() {
            bail!("Self-hosted integrations must specify an API base");
        }

        let id = self
            .db
            .create_integration(
                kind.as_enum_str().to_string(),
                display_name,
                access_token,
                api_base,
            )
            .await?;
        let id = id.as_id();
        let _ = self
            .job
            .trigger(BackgroundJobEvent::SyncThirdPartyRepositories(id.clone()).to_command())
            .await;
        Ok(id)
    }

    async fn delete_integration(&self, id: ID, kind: IntegrationKind) -> Result<()> {
        self.db
            .delete_integration(id.as_rowid()?, kind.as_enum_str())
            .await?;
        self.job
            .clear(BackgroundJobEvent::SyncThirdPartyRepositories(id).to_command())
            .await?;
        self.job
            .trigger(BackgroundJobEvent::IndexGarbageCollection.to_command())
            .await?;
        Ok(())
    }

    async fn update_integration(
        &self,
        id: ID,
        kind: IntegrationKind,
        display_name: String,
        access_token: Option<String>,
        api_base: Option<String>,
    ) -> Result<()> {
        if kind.is_self_hosted() && api_base.is_none() {
            bail!("Self-hosted integrations must specify an API base");
        }

        let integration = self.get_integration(id.clone()).await?;
        let access_token_is_changed = access_token
            .as_ref()
            .is_some_and(|token| token != &integration.access_token);
        let api_base_is_changed = integration.api_base != api_base;

        self.db
            .update_integration(
                id.as_rowid()?,
                kind.as_enum_str(),
                display_name,
                access_token,
                api_base,
            )
            .await?;

        if access_token_is_changed || api_base_is_changed {
            let _ = self
                .job
                .trigger(BackgroundJobEvent::SyncThirdPartyRepositories(id.clone()).to_command())
                .await;
        }

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
    use crate::job;

    async fn create_services() -> (Arc<dyn IntegrationService>, Arc<dyn JobService>, DbConn) {
        let db = DbConn::new_in_memory().await.unwrap();
        let job = Arc::new(job::create(db.clone()).await);
        (Arc::new(create(db.clone(), job.clone())), job, db)
    }

    #[tokio::test]
    async fn test_integration_crud() {
        let (integration, _, _) = create_services().await;
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
            .update_integration(
                id.clone(),
                IntegrationKind::Gitlab,
                "id2".into(),
                None,
                None,
            )
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

    #[tokio::test]
    async fn test_update_integration_should_reset_status() {
        let (integration, _, db) = create_services().await;
        let id = integration
            .create_integration(IntegrationKind::Github, "gh".into(), "token".into(), None)
            .await
            .unwrap();

        // Test event is sent to re-sync provider after provider is created
        let job = db.get_next_job_to_execute().await.unwrap();
        assert_eq!(
            job.command,
            BackgroundJobEvent::SyncThirdPartyRepositories(id.clone()).to_command()
        );

        // Test integration status is failed after updating sync status with an error
        integration
            .update_integration_sync_status(id.clone(), Some("error".into()))
            .await
            .unwrap();

        let provider = integration.get_integration(id.clone()).await.unwrap();

        assert_eq!(provider.status, IntegrationStatus::Failed);

        // Test integration status is not changed if token has not been updated
        integration
            .update_integration(
                id.clone(),
                IntegrationKind::Github,
                "gh".into(),
                Some("token".into()),
                None,
            )
            .await
            .unwrap();

        let provider = integration.get_integration(id.clone()).await.unwrap();

        assert_eq!(provider.status, IntegrationStatus::Failed);

        // Test integration status is pending after updating token
        integration
            .update_integration(
                id.clone(),
                IntegrationKind::Github,
                "gh".into(),
                Some("token2".into()),
                None,
            )
            .await
            .unwrap();

        let provider = integration.get_integration(id.clone()).await.unwrap();

        assert_eq!(provider.status, IntegrationStatus::Pending);

        // Test integration status is ready after a successful sync and an update which changes no fields
        integration
            .update_integration_sync_status(id.clone(), None)
            .await
            .unwrap();
        integration
            .update_integration(id.clone(), IntegrationKind::Github, "gh".into(), None, None)
            .await
            .unwrap();

        let provider = integration.get_integration(id.clone()).await.unwrap();

        assert_eq!(provider.status, IntegrationStatus::Ready);

        // Test event is sent to re-sync provider after credentials are updated
        let job = db.get_next_job_to_execute().await.unwrap();
        assert_eq!(
            job.command,
            BackgroundJobEvent::SyncThirdPartyRepositories(id.clone()).to_command()
        );

        // Test sync event is not sent if no fields are updated
        integration
            .update_integration(
                id.clone(),
                IntegrationKind::Github,
                "gh".into(),
                Some("token2".into()),
                None,
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_api_base() {
        let (integration, _, _) = create_services().await;

        // Should not be able to create a self-hosted provider without an API base
        assert!(integration
            .create_integration(
                IntegrationKind::GithubSelfHosted,
                "github".into(),
                "token".into(),
                None
            )
            .await
            .is_err());

        // Should not be able to update an existing self-hosted provider without an API base
        let id = integration
            .create_integration(
                IntegrationKind::GitlabSelfHosted,
                "github".into(),
                "token".into(),
                Some("https://github.com".into()),
            )
            .await
            .unwrap();

        assert!(integration
            .update_integration(
                id,
                IntegrationKind::GitlabSelfHosted,
                "github".into(),
                None,
                None
            )
            .await
            .is_err());
    }
}
