mod db;
mod helper;
mod scheduler;
mod third_party_integration;

use std::{str::FromStr, sync::Arc};

use cron::Schedule;
use futures::StreamExt;
use helper::{CronStream, Job, JobLogger};
use juniper::ID;
use tabby_common::config::{ConfigAccess, RepositoryConfig};
use tabby_db::DbConn;
use tabby_inference::Embedding;
use tabby_schema::{integration::IntegrationService, repository::ThirdPartyRepositoryService};
use third_party_integration::IndexIssuesJob;
use tracing::warn;

use self::{
    db::DbMaintainanceJob, scheduler::SchedulerJob, third_party_integration::SyncIntegrationJob,
};

#[derive(PartialEq, Debug)]
pub enum BackgroundJobEvent {
    Scheduler(RepositoryConfig),
    SyncThirdPartyRepositories(ID),
    IndexIssues(ID),
}

pub async fn start(
    db: DbConn,
    config_access: Arc<dyn ConfigAccess>,
    third_party_repository_service: Arc<dyn ThirdPartyRepositoryService>,
    integration_service: Arc<dyn IntegrationService>,
    embedding: Arc<dyn Embedding>,
    sender: tokio::sync::mpsc::UnboundedSender<BackgroundJobEvent>,
    mut receiver: tokio::sync::mpsc::UnboundedReceiver<BackgroundJobEvent>,
) {
    let mut hourly =
        CronStream::new(Schedule::from_str("@hourly").expect("Invalid cron expression"))
            .into_stream();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                Some(event) = receiver.recv() => {
                    match event {
                        BackgroundJobEvent::Scheduler(repository_config) => {
                            let job = SchedulerJob::new(repository_config);
                            let mut job_logger = JobLogger::new(SchedulerJob::NAME, db.clone()).await;
                            if let Err(err) = job.run(job_logger.clone(), embedding.clone()).await {
                                cprintln!(job_logger, "{:?}", err);
                                job_logger.complete(-1).await;
                            } else {
                                job_logger.complete(0).await;
                            }
                        },
                        BackgroundJobEvent::SyncThirdPartyRepositories(integration_id) => {
                            let job = SyncIntegrationJob::new(integration_id);
                            if let Err(err) = job.run(third_party_repository_service.clone()).await {
                                warn!("Sync integration job failed: {err:?}");
                            }
                        }
                        BackgroundJobEvent::IndexIssues(integration_id) => {
                            let job = IndexIssuesJob::new(integration_id);
                            if let Err(err) = job.run(embedding.clone(), third_party_repository_service.clone(), integration_service.clone()).await {
                                warn!("Index issues job failed: {err:?}");
                            }
                        }
                    }
                },
                Some(now) = hourly.next() => {
                    if let Err(err) = DbMaintainanceJob::cron(now, db.clone()).await {
                        warn!("Database maintainance failed: {:?}", err);
                    }

                    if let Err(err) = SchedulerJob::cron(now, config_access.clone(), sender.clone()).await {
                        warn!("Scheduler job failed: {:?}", err);
                    }

                    if let Err(err) = SyncIntegrationJob::cron(now, sender.clone(), integration_service.clone()).await {
                        warn!("Sync integration job failed: {:?}", err);
                    }

                    if let Err(err) = IndexIssuesJob::cron(now, sender.clone(), integration_service.clone()).await {
                        warn!("Index issues job failed: {err:?}");
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

macro_rules! cprintln {
    ($ctx:expr, $($params:tt)+) => {
        {
            tracing::debug!($($params)+);
            $ctx.r#internal_println(format!($($params)+)).await;
        }
    }
}

use cprintln;
