use std::sync::Arc;

use apalis::{
    prelude::{Data, Job, Monitor, Storage, WorkerFactoryFn},
    sqlite::{SqlitePool, SqliteStorage},
    utils::TokioExecutor,
};
use chrono::{DateTime, Utc};
use juniper::ID;
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;
use tabby_schema::{integration::IntegrationService, repository::ThirdPartyRepositoryService};
use tracing::debug;

use super::{helper::BasicJob, helper::CronJob};

#[derive(Serialize, Deserialize, Clone)]
pub struct ThirdPartyRepositorySyncJob {
    integration_id: ID,
}

impl Job for ThirdPartyRepositorySyncJob {
    const NAME: &'static str = "third_party_repository_sync";
}

impl CronJob for ThirdPartyRepositorySyncJob {
    const SCHEDULE: &'static str = "@hourly";
}

impl ThirdPartyRepositorySyncJob {
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
        storage: Data<SqliteStorage<ThirdPartyRepositorySyncJob>>,
        integration_service: Data<Arc<dyn IntegrationService>>,
    ) -> tabby_schema::Result<()> {
        debug!("Syncing all third-party repositories");

        let mut storage = (*storage).clone();
        for integration in integration_service
            .list_integrations(None, None, None, None, None, None)
            .await?
        {
            storage
                .push(ThirdPartyRepositorySyncJob::new(integration.id))
                .await
                .expect("Unable to push job");
        }
        Ok(())
    }

    pub fn register(
        monitor: Monitor<TokioExecutor>,
        pool: SqlitePool,
        db: DbConn,
        repository_service: Arc<dyn ThirdPartyRepositoryService>,
        integration_service: Arc<dyn IntegrationService>,
    ) -> (
        SqliteStorage<ThirdPartyRepositorySyncJob>,
        Monitor<TokioExecutor>,
    ) {
        let storage = SqliteStorage::new(pool);
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
                    .build_fn(ThirdPartyRepositorySyncJob::cron),
            );
        (storage, monitor)
    }
}
