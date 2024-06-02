mod db;
mod helper;
mod scheduler;
mod third_party_integration;

use std::{str::FromStr, sync::Arc};

use cron::Schedule;
use futures::StreamExt;
use helper::{CronStream, Job, JobLogger, JobQueue};
use juniper::ID;
use tabby_common::config::{ConfigAccess, RepositoryConfig};
use tabby_db::DbConn;
use tabby_inference::Embedding;
use tabby_schema::{integration::IntegrationService, repository::ThirdPartyRepositoryService};
use tracing::warn;

use self::{
    db::DbMaintainanceJob, scheduler::SchedulerJob, third_party_integration::SyncIntegrationJob,
};

pub enum BackgroundJobEvent {
    Scheduler(RepositoryConfig),
    SyncThirdPartyRepositories(ID),
}

struct BackgroundJobImpl {
    scheduler: JobQueue<SchedulerJob>,
    third_party_repository: JobQueue<SyncIntegrationJob>,
}

pub async fn start(
    db: DbConn,
    config_access: Arc<dyn ConfigAccess>,
    third_party_repository_service: Arc<dyn ThirdPartyRepositoryService>,
    integration_service: Arc<dyn IntegrationService>,
    embedding: Arc<dyn Embedding>,
    mut receiver: tokio::sync::mpsc::UnboundedReceiver<BackgroundJobEvent>,
) {
    let mut hourly =
        CronStream::new(Schedule::from_str("1 * * * * *").expect("Invalid cron expression"))
            .into_stream();

    tokio::spawn(async move {
        let (tx, mut scheduler_rx) = tokio::sync::mpsc::unbounded_channel();
        let scheduler = JobQueue::new(tx);

        let (tx, mut third_party_repository_rx) = tokio::sync::mpsc::unbounded_channel();
        let third_party_repository = JobQueue::new(tx);

        let mut background_job = BackgroundJobImpl {
            scheduler,
            third_party_repository,
        };

        loop {
            tokio::select! {
                Some(event) = receiver.recv() => {
                    background_job.on_event_publish(event).await;
                },
                Some(job) = scheduler_rx.recv() => {
                    let mut job_logger = JobLogger::new(SchedulerJob::NAME, db.clone()).await;
                    if let Err(err) = job.run(job_logger.clone(), embedding.clone()).await {
                        cprintln!(job_logger, "{:?}", err);
                        job_logger.complete(-1).await;
                    } else {
                        job_logger.complete(0).await;
                    }
                },
                Some(job) = third_party_repository_rx.recv() => {
                    if let Err(err) = job.run(third_party_repository_service.clone()).await {
                        warn!("Sync integration job failed: {:?}", err);
                    }
                },
                Some(now) = hourly.next() => {
                    if let Err(err) = DbMaintainanceJob::cron(now, db.clone()).await {
                        warn!("Database maintainance failed: {:?}", err);
                    }

                    if let Err(err) = SchedulerJob::cron(now, config_access.clone(), background_job.scheduler.clone()).await {
                        warn!("Scheduler job failed: {:?}", err);
                    }

                    if let Err(err) = SyncIntegrationJob::cron(now, background_job.third_party_repository.clone(), integration_service.clone()).await {
                        warn!("Sync integration job failed: {:?}", err);
                    }
                },
                else => {
                    warn!("Background job channel closed");
                    break;
                }
            };
        }
    });
}

impl BackgroundJobImpl {
    async fn trigger_scheduler(&self, repository: RepositoryConfig) {
        self.scheduler
            .enqueue(SchedulerJob::new(repository))
            .expect("unable to push job");
    }

    async fn trigger_sync_integration(&self, provider_id: ID) {
        self.third_party_repository
            .enqueue(SyncIntegrationJob::new(provider_id))
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
