mod db;
mod helper;
mod scheduler;
mod third_party_integration;

use std::{sync::Arc, time::Duration};

use apalis::{
    prelude::{Monitor, Storage},
    sqlite::{SqlitePool, SqliteStorage},
};
use juniper::ID;
use tabby_common::config::{ConfigAccess, RepositoryConfig};
use tabby_db::DbConn;
use tabby_schema::{integration::IntegrationService, repository::ThirdPartyRepositoryService};

use self::{
    db::DbMaintainanceJob, scheduler::SchedulerJob, third_party_integration::SyncIntegrationJob,
};
use crate::path::job_db_file;

pub enum BackgroundJobEvent {
    Scheduler(RepositoryConfig),
    SyncThirdPartyRepositories(ID),
}

struct BackgroundJobImpl {
    scheduler: SqliteStorage<SchedulerJob>,
    third_party_repository: SqliteStorage<SyncIntegrationJob>,
}

pub async fn start(
    db: DbConn,
    config_access: Arc<dyn ConfigAccess>,
    third_party_repository_service: Arc<dyn ThirdPartyRepositoryService>,
    integration_service: Arc<dyn IntegrationService>,
    mut receiver: tokio::sync::mpsc::UnboundedReceiver<BackgroundJobEvent>,
) {
    let path = format!("sqlite://{}?mode=rwc", job_db_file().display());
    let pool = SqlitePool::connect(&path)
        .await
        .expect("unable to create sqlite pool");
    SqliteStorage::setup(&pool)
        .await
        .expect("unable to run migrations for sqlite");

    let config = apalis_sql::Config::default().poll_interval(Duration::from_secs(5));
    let monitor = Monitor::new();
    let monitor = DbMaintainanceJob::register(monitor, db.clone());
    let (scheduler, monitor) = SchedulerJob::register(
        monitor,
        pool.clone(),
        db.clone(),
        config.clone(),
        config_access,
    );
    let (third_party_repository, monitor) = SyncIntegrationJob::register(
        monitor,
        pool.clone(),
        db.clone(),
        config.clone(),
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
            .push(SchedulerJob::new(repository))
            .await
            .expect("unable to push job");
    }

    async fn trigger_sync_integration(&self, provider_id: ID) {
        self.third_party_repository
            .clone()
            .push(SyncIntegrationJob::new(provider_id))
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
