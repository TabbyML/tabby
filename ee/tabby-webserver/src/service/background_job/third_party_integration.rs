use std::sync::Arc;

use anyhow::anyhow;
use chrono::{DateTime, Utc};
use juniper::ID;
use serde::{Deserialize, Serialize};
use tabby_schema::{integration::IntegrationService, repository::ThirdPartyRepositoryService};
use tracing::debug;

use super::{helper::Job, BackgroundJobEvent};

#[derive(Serialize, Deserialize, Clone)]
pub struct SyncIntegrationJob {
    integration_id: ID,
}

impl Job for SyncIntegrationJob {
    const NAME: &'static str = "third_party_repository_sync";
}

impl SyncIntegrationJob {
    pub fn new(integration_id: ID) -> Self {
        Self { integration_id }
    }

    pub async fn run(
        self,
        repository_service: Arc<dyn ThirdPartyRepositoryService>,
    ) -> tabby_schema::Result<()> {
        repository_service
            .sync_repositories(self.integration_id)
            .await?;
        Ok(())
    }

    pub async fn cron(
        _now: DateTime<Utc>,
        sender: tokio::sync::mpsc::UnboundedSender<BackgroundJobEvent>,
        integration_service: Arc<dyn IntegrationService>,
    ) -> tabby_schema::Result<()> {
        debug!("Syncing all third-party repositories");

        for integration in integration_service
            .list_integrations(None, None, None, None, None, None)
            .await?
        {
            sender
                .send(BackgroundJobEvent::SyncThirdPartyRepositories(
                    integration.id,
                ))
                .map_err(|_| anyhow!("Failed to enqueue scheduler job"))?;
        }
        Ok(())
    }
}
