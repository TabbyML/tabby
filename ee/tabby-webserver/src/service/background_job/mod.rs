mod daily;
mod db;
mod git;
mod helper;
mod hourly;
mod index_garbage_collection;
mod license_check;
mod third_party_integration;
mod web_crawler;

use std::{str::FromStr, sync::Arc};

use cron::Schedule;
use daily::DailyJob;
use futures::StreamExt;
use git::SchedulerGitJob;
use helper::{CronStream, Job, JobLogger};
use hourly::HourlyJob;
use index_garbage_collection::IndexGarbageCollection;
use juniper::ID;
use license_check::LicenseCheckJob;
use serde::{Deserialize, Serialize};
use tabby_common::config::CodeRepository;
use tabby_db::DbConn;
use tabby_inference::Embedding;
use tabby_schema::{
    context::ContextService,
    integration::IntegrationService,
    job::JobService,
    license::LicenseService,
    notification::{NotificationRecipient, NotificationService},
    repository::{GitRepositoryService, RepositoryService, ThirdPartyRepositoryService},
    AsID,
};
use third_party_integration::SchedulerGithubGitlabJob;
use tracing::{debug, warn};
pub use web_crawler::WebCrawlerJob;

use self::third_party_integration::SyncIntegrationJob;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum BackgroundJobEvent {
    SchedulerGitRepository(CodeRepository),
    SchedulerGithubGitlabRepository(ID),
    SyncThirdPartyRepositories(ID),
    WebCrawler(WebCrawlerJob),
    IndexGarbageCollection,
    Hourly,
    Daily,
}

impl BackgroundJobEvent {
    pub fn name(&self) -> &'static str {
        match self {
            BackgroundJobEvent::SchedulerGitRepository(_) => SchedulerGitJob::NAME,
            BackgroundJobEvent::SchedulerGithubGitlabRepository(_) => {
                SchedulerGithubGitlabJob::NAME
            }
            BackgroundJobEvent::SyncThirdPartyRepositories(_) => SyncIntegrationJob::NAME,
            BackgroundJobEvent::WebCrawler(_) => WebCrawlerJob::NAME,
            BackgroundJobEvent::IndexGarbageCollection => IndexGarbageCollection::NAME,
            BackgroundJobEvent::Hourly => HourlyJob::NAME,
            BackgroundJobEvent::Daily => DailyJob::NAME,
        }
    }

    pub fn to_command(&self) -> String {
        serde_json::to_string(self).expect("Failed to serialize background job event")
    }
}

fn background_job_notification_name(event: &BackgroundJobEvent) -> &str {
    match event {
        BackgroundJobEvent::SchedulerGitRepository(_) => "Indexing Repository",
        BackgroundJobEvent::SchedulerGithubGitlabRepository(_) => "Indexing Repository",
        BackgroundJobEvent::SyncThirdPartyRepositories(_) => "Loading Repository",
        BackgroundJobEvent::WebCrawler(_) => "Web Indexing",
        BackgroundJobEvent::IndexGarbageCollection => "Garbage Collection",
        BackgroundJobEvent::Hourly => "Hourly",
        BackgroundJobEvent::Daily => "Daily",
    }
}

async fn notify_job_error(
    notification_service: Arc<dyn NotificationService>,
    err: &str,
    event: &BackgroundJobEvent,
    id: i64,
) {
    warn!("job {:?} failed: {:?}", event, err);
    let name = background_job_notification_name(event);
    if let Err(err) = notification_service
        .create(
            NotificationRecipient::Admin,
            &format!(
                r#"Job **{}** has failed.

Please examine the [logs](/jobs/detail?id={}) to determine the underlying issue.
"#,
                name,
                id.as_id()
            ),
        )
        .await
    {
        warn!("Failed to send notification: {:?}", err);
    }
}

pub async fn start(
    db: DbConn,
    job_service: Arc<dyn JobService>,
    git_repository_service: Arc<dyn GitRepositoryService>,
    third_party_repository_service: Arc<dyn ThirdPartyRepositoryService>,
    integration_service: Arc<dyn IntegrationService>,
    repository_service: Arc<dyn RepositoryService>,
    context_service: Arc<dyn ContextService>,
    license_service: Arc<dyn LicenseService>,
    notification_service: Arc<dyn NotificationService>,
    embedding: Arc<dyn Embedding>,
) {
    let mut hourly =
        CronStream::new(Schedule::from_str("@hourly").expect("Invalid cron expression"))
            .into_stream();

    let mut daily = CronStream::new(Schedule::from_str("@daily").expect("Invalid cron expression"))
        .into_stream();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                job = db.get_next_job_to_execute() => {
                    let Some(job) = job else {
                        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                        continue;
                    };

                    if let Err(err) = db.update_job_started(job.id).await {
                        warn!("Failed to mark job status to started: {:?}", err);
                        continue;
                    }

                    let logger = JobLogger::new(db.clone(), job.id);
                    debug!("Background job {} started, command: {}", job.id, job.command);
                    let Ok(event) = serde_json::from_str::<BackgroundJobEvent>(&job.command) else {
                        logkit::info!(exit_code = -1; "Failed to parse background job event, marking it as failed");
                        continue;
                    };

                    let cloned_event = event.clone();
                    if let Err(err) = match event {
                        BackgroundJobEvent::SchedulerGitRepository(repository_config) => {
                            let job = SchedulerGitJob::new(repository_config);
                            job.run(embedding.clone()).await
                        },
                        BackgroundJobEvent::SyncThirdPartyRepositories(integration_id) => {
                            let job = SyncIntegrationJob::new(integration_id);
                            job.run(third_party_repository_service.clone()).await
                        }
                        BackgroundJobEvent::SchedulerGithubGitlabRepository(integration_id) => {
                            let job = SchedulerGithubGitlabJob::new(integration_id);
                            job.run(embedding.clone(), third_party_repository_service.clone(), integration_service.clone()).await
                        }
                        BackgroundJobEvent::WebCrawler(job) => {
                            job.run(embedding.clone()).await
                        }
                        BackgroundJobEvent::IndexGarbageCollection => {
                            let job = IndexGarbageCollection;
                            job.run(repository_service.clone(), context_service.clone()).await
                        }
                        BackgroundJobEvent::Hourly => {
                            let job = HourlyJob;
                            job.run(
                                db.clone(),
                                context_service.clone(),
                                git_repository_service.clone(),
                                job_service.clone(),
                                integration_service.clone(),
                                third_party_repository_service.clone(),
                                repository_service.clone(),
                            ).await
                        }
                        BackgroundJobEvent::Daily => {
                            let job = DailyJob;
                            job.run(
                                license_service.clone(),
                                notification_service.clone(),
                            ).await
                        }
                    } {
                        logkit::info!(exit_code = 1; "Job failed {}", err);
                        notify_job_error(notification_service.clone(), &err.to_string(), &cloned_event, job.id).await;
                    } else {
                        logkit::info!(exit_code = 0; "Job completed successfully");
                    }
                    logger.finalize().await;
                    debug!("Background job {} completed", job.id);
                },
                Some(_) = hourly.next() => {
                    match job_service.trigger(BackgroundJobEvent::Hourly.to_command()).await {
                        Err(err) => warn!("Hourly background job schedule failed {}", err),
                        Ok(id) => debug!("Hourly background job {} scheduled", id),
                    }
                },
                Some(_) = daily.next() => {
                    match job_service.trigger(BackgroundJobEvent::Daily.to_command()).await {
                        Err(err) => warn!("Daily background job schedule failed {}", err),
                        Ok(id) => debug!("Daily background job {} scheduled", id),
                    }
                }
                else => {
                    warn!("Background job channel closed");
                    break;
                }
            };
        }
    });
}
