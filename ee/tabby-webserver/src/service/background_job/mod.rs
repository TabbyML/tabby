mod db;
mod helper;
mod scheduler;
mod third_party_integration;

use std::sync::Arc;

use apalis::prelude::{MemoryStorage, MessageQueue, Monitor};
use juniper::ID;
use tabby_common::config::{ConfigAccess, RepositoryConfig};
use tabby_db::DbConn;
use tabby_inference::Embedding;
use tabby_schema::{integration::IntegrationService, repository::ThirdPartyRepositoryService};

use self::{
    db::DbMaintainanceJob, scheduler::SchedulerJob, third_party_integration::SyncIntegrationJob,
};

pub enum BackgroundJobEvent {
    Scheduler(RepositoryConfig),
    SyncThirdPartyRepositories(ID),
}

struct BackgroundJobImpl {
    scheduler: MemoryStorage<SchedulerJob>,
    third_party_repository: MemoryStorage<SyncIntegrationJob>,
}

pub async fn start(
    db: DbConn,
    config_access: Arc<dyn ConfigAccess>,
    third_party_repository_service: Arc<dyn ThirdPartyRepositoryService>,
    integration_service: Arc<dyn IntegrationService>,
    embedding: Arc<dyn Embedding>,
    mut receiver: tokio::sync::mpsc::UnboundedReceiver<BackgroundJobEvent>,
) {
    let monitor = Monitor::new();
    let monitor = DbMaintainanceJob::register(monitor, db.clone());
    let (scheduler, monitor) =
        SchedulerJob::register(monitor, db.clone(), config_access, embedding);
    let (third_party_repository, monitor) = SyncIntegrationJob::register(
        monitor,
        db.clone(),
        third_party_repository_service,
        integration_service,
    );

    tokio::spawn(async move {
        monitor.run().await.expect("failed to start worker");
    });

    tokio::spawn(async move {
        let mut background_job = BackgroundJobImpl {
            scheduler,
            third_party_repository,
        };

        while let Some(event) = receiver.recv().await {
            background_job.on_event_publish(event).await;
        }
    });
}

impl BackgroundJobImpl {
    async fn trigger_scheduler(&self, repository: RepositoryConfig) {
        self.scheduler
            .clone()
            .enqueue(SchedulerJob::new(repository))
            .await
            .expect("unable to push job");
    }

    async fn trigger_sync_integration(&self, provider_id: ID) {
        self.third_party_repository
            .clone()
            .enqueue(SyncIntegrationJob::new(provider_id))
            .await
            .expect("Unable to push job");
    }

    async fn on_event_publish(&mut self, event: BackgroundJobEvent) {
        match event {
            BackgroundJobEvent::Scheduler(repository) => self.trigger_scheduler(repository).await,
            BackgroundJobEvent::SyncThirdPartyRepositories(integration_id) => {
                self.trigger_sync_integration(integration_id).await
            }
        }
    }
}

macro_rules! cprintln {
    ($ctx:expr, $($params:tt)+) => {
        {
            tracing::debug!($($params)+);
            $ctx.r#internal_println(format!($($params)+)).await;
        }
    }
}

use cprintln;
