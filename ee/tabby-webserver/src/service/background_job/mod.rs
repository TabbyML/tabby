mod db;
mod git;
mod helper;
mod index_garbage_collection;
mod license_check;
mod third_party_integration;
mod web_crawler;

use std::{fmt::Display, str::FromStr, sync::Arc};

use cron::Schedule;
use futures::StreamExt;
use git::SchedulerGitJob;
use helper::{CronStream, Job, JobLogger};
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

use self::{db::DbMaintainanceJob, third_party_integration::SyncIntegrationJob};

#[derive(Debug, Serialize, Deserialize)]
pub enum BackgroundJobEvent {
    SchedulerGitRepository(CodeRepository),
    SchedulerGithubGitlabRepository(ID),
    SyncThirdPartyRepositories(ID),
    WebCrawler(WebCrawlerJob),
    IndexGarbageCollection,
}

impl Display for BackgroundJobEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackgroundJobEvent::SchedulerGitRepository(repository) => {
                write!(f, "SyncGitRepository::{}", repository.git_url)
            }
            BackgroundJobEvent::SchedulerGithubGitlabRepository(integration_id) => {
                write!(f, "SyncGithubGitlabRepository::{}", integration_id)
            }
            BackgroundJobEvent::SyncThirdPartyRepositories(integration_id) => {
                write!(f, "SyncThirdPartyRepositories::{}", integration_id)
            }
            BackgroundJobEvent::WebCrawler(job) => write!(f, "WebCrawler::{}", job.url()),
            BackgroundJobEvent::IndexGarbageCollection => write!(f, "IndexGarbageCollection"),
        }
    }
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
        }
    }

    pub fn to_command(&self) -> String {
        serde_json::to_string(self).expect("Failed to serialize background job event")
    }
}

#[macro_export]
macro_rules! notify_job_error {
    ($notification_service:expr, $err:expr, $name:expr, $id:expr) => {{
        let id = $id.as_id();
        warn!("job {} failed: {:?}", $name, $err);
        $notification_service
            .create(
                NotificationRecipient::Admin,
                &format!(
                    r#"Background job failed

Job `{}` has failed.

Please check the log at [Jobs Detail](/jobs/detail?id={}) to identify the underlying issue.
"#,
                    $name, id
                ),
            )
            .await
            .unwrap();
    }};
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

                    let job_id = job.id;
                    let logger = JobLogger::new(db.clone(), job_id);
                    debug!("Background job {} started, command: {}", job_id, job.command);
                    let Ok(event) = serde_json::from_str::<BackgroundJobEvent>(&job.command) else {
                        logkit::info!(exit_code = -1; "Failed to parse background job event, marking it as failed");
                        continue;
                    };

                    let job_name = event.to_string();
                    let result = match event {
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
                            job.run(repository_service.clone(), context_service.clone(), db.clone(), job_id).await
                        }
                    };
                    debug!("Background job {} completed", job.id);

                    match &result {
                        Err(err) => {
                            logkit::info!(exit_code = 1; "Job failed {}", err);
                            logger.finalize().await;
                            notify_job_error!(notification_service, err, job_name, job_id);
                        },
                        _ => {
                            logkit::info!(exit_code = 0; "Job completed successfully");
                            logger.finalize().await;
                        }
                    }
                },
                Some(now) = hourly.next() => {
                    let job_id = match db.create_job_run(DbMaintainanceJob.name().to_string(), DbMaintainanceJob.to_command()).await {
                        Ok(job_id) => job_id,
                        Err(_) => {
                            warn!("Failed to create job run");
                            continue;
                        }
                    };
                    if let Err(err) = DbMaintainanceJob::cron(now, context_service.clone(), db.clone(), job_id).await {
                        notify_job_error!(notification_service, err, DbMaintainanceJob.name(), job_id);
                    }

                    let job_id = match db.create_job_run(SchedulerGitJob::NAME.to_string(), "cron".to_string()).await {
                        Ok(job_id) => job_id,
                        Err(_) => {
                            warn!("Failed to create job run");
                            continue;
                        }
                    };
                    if let Err(err) = SchedulerGitJob::cron(now, git_repository_service.clone(), job_service.clone(), db.clone(), job_id).await {
                        notify_job_error!(notification_service, err, SchedulerGitJob::NAME, job_id);
                    }

                    let job_id = match db.create_job_run(SyncIntegrationJob::NAME.to_string(), "cron".to_string()).await {
                        Ok(job_id) => job_id,
                        Err(_) => {
                            warn!("Failed to create job run");
                            continue;
                        }
                    };
                    if let Err(err) = SyncIntegrationJob::cron(now, integration_service.clone(), job_service.clone(), db.clone(), job_id).await {
                        notify_job_error!(notification_service, err, SyncIntegrationJob::NAME, job_id);
                    }

                    let job_id = match db.create_job_run(SchedulerGithubGitlabJob::NAME.to_string(), "cron".to_string()).await {
                        Ok(job_id) => job_id,
                        Err(_) => {
                            warn!("Failed to create job run");
                            continue;
                        }
                    };
                    if let Err(err) = SchedulerGithubGitlabJob::cron(now, third_party_repository_service.clone(), job_service.clone(), db.clone(), job_id).await {
                        notify_job_error!(notification_service, err, SchedulerGithubGitlabJob::NAME, job_id);
                    }

                    let job_id = match db.create_job_run(IndexGarbageCollection.name().to_string(), IndexGarbageCollection.to_command()).await {
                        Ok(job_id) => job_id,
                        Err(_) => {
                            warn!("Failed to create job run");
                            continue;
                        }
                    };
                    if let Err(err) = IndexGarbageCollection.run(repository_service.clone(), context_service.clone(), db.clone(), job_id).await {
                        notify_job_error!(notification_service, err, IndexGarbageCollection.name(), job_id);
                    }
                },
                Some(now) = daily.next() => {
                    let job_id = match db.create_job_run(LicenseCheckJob::NAME.to_string(), "cron".to_string()).await {
                        Ok(job_id) => job_id,
                        Err(_) => {
                            warn!("Failed to create job run");
                            continue;
                        }
                    };
                    if let Err(err) = LicenseCheckJob::cron(now, license_service.clone(), notification_service.clone(), db.clone(), job_id).await {
                        notify_job_error!(notification_service, err, LicenseCheckJob::NAME, job_id);
                    }
                }
                else => {
                    warn!("Background job channel closed");
                    return;
                }
            };
        }
    });
}
