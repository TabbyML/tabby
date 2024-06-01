use std::sync::Arc;

use anyhow::anyhow;
use apalis::{
    prelude::{Data, Job, MemoryStorage, MessageQueue, Monitor, WorkerFactoryFn},
    utils::TokioExecutor,
};
use chrono::{DateTime, Utc};
use juniper::ID;
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;
use tabby_schema::{integration::IntegrationService, repository::ThirdPartyRepositoryService};
use tracing::debug;

use super::helper::{BasicJob, CronJob};

#[derive(Serialize, Deserialize, Clone)]
pub struct SyncIntegrationJob {
    integration_id: ID,
}

impl Job for SyncIntegrationJob {
    const NAME: &'static str = "third_party_repository_sync";
}

impl CronJob for SyncIntegrationJob {
    const SCHEDULE: &'static str = "@hourly";
}

impl SyncIntegrationJob {
    pub fn new(integration_id: ID) -> Self {
        Self { integration_id }
    }

    async fn run(
        self,
        repository_service: Data<Arc<dyn ThirdPartyRepositoryService>>,
    ) -> tabby_schema::Result<()> {
        repository_service
            .sync_repositories(self.integration_id)
            .await?;
        Ok(())
    }

    async fn cron(
        _now: DateTime<Utc>,
        storage: Data<MemoryStorage<SyncIntegrationJob>>,
        integration_service: Data<Arc<dyn IntegrationService>>,
    ) -> tabby_schema::Result<()> {
        debug!("Syncing all third-party repositories");

        for integration in integration_service
            .list_integrations(None, None, None, None, None, None)
            .await?
        {
            storage
                .enqueue(SyncIntegrationJob::new(integration.id))
                .await
                .map_err(|_| anyhow!("Failed to enqueue scheduler job"))?;
        }
        Ok(())
    }

    pub fn register(
        monitor: Monitor<TokioExecutor>,
        db: DbConn,
        repository_service: Arc<dyn ThirdPartyRepositoryService>,
        integration_service: Arc<dyn IntegrationService>,
    ) -> (MemoryStorage<SyncIntegrationJob>, Monitor<TokioExecutor>) {
        let storage = MemoryStorage::default();
        let monitor = monitor
            .register(
                Self::basic_worker(storage.clone(), db.clone())
                    .data(repository_service.clone())
                    .build_fn(Self::run),
            )
            .register(
                Self::cron_worker(db)
                    .data(integration_service)
                    .data(storage.clone())
                    .build_fn(SyncIntegrationJob::cron),
            );
        (storage, monitor)
    }
}
