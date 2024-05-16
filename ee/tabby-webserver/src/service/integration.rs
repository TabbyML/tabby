use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    integration::{IntegrationAccessToken, IntegrationKind, IntegrationService},
    AsID, AsRowid, DbEnum, Result,
};
use tokio::sync::mpsc::UnboundedSender;

use crate::service::background_job::BackgroundJobEvent;

use super::graphql_pagination_to_filter;

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
    ) -> Result<ID> {
        let id = self
            .db
            .create_integration_access_token(
                kind.as_enum_str().to_string(),
                display_name,
                access_token,
            )
            .await?;
        let id = id.as_id();
        let _ = self
            .background_job
            .send(BackgroundJobEvent::SyncThirdPartyRepositories(id.clone()));
        Ok(id)
    }

    async fn delete_integration(&self, id: ID) -> Result<()> {
        self.db
            .delete_integration_access_token(id.as_rowid()?)
            .await?;
        Ok(())
    }

    async fn update_integration(
        &self,
        id: ID,
        display_name: String,
        access_token: Option<String>,
    ) -> Result<()> {
        self.db
            .update_integration_access_token(id.as_rowid()?, display_name, access_token)
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
    ) -> Result<Vec<IntegrationAccessToken>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let ids = ids
            .unwrap_or_default()
            .into_iter()
            .map(|id| id.as_rowid())
            .collect::<Result<_, _>>()?;
        let kind = kind.map(|kind| kind.as_enum_str().to_string());
        let integrations = self
            .db
            .list_integration_access_tokens(ids, kind, limit, skip_id, backwards)
            .await?;

        Ok(integrations
            .into_iter()
            .map(IntegrationAccessToken::try_from)
            .collect::<Result<_, _>>()?)
    }

    async fn get_integration(&self, id: ID) -> Result<IntegrationAccessToken> {
        Ok(self
            .db
            .get_integration_access_token(id.as_rowid()?)
            .await?
            .try_into()?)
    }

    async fn update_integration_error(&self, id: ID, error: Option<String>) -> Result<()> {
        self.db
            .update_integration_access_token_error(id.as_rowid()?, error)
            .await?;
        Ok(())
    }
}
