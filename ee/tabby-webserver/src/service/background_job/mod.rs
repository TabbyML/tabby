mod db;
mod git;
mod helper;
mod third_party_integration;
mod web_crawler;

use std::{str::FromStr, sync::Arc};

use cron::Schedule;
use futures::StreamExt;
use git::SchedulerGitJob;
pub use helper::Job;
use helper::{CronStream, JobLogger};
use juniper::ID;
use serde::{Deserialize, Serialize};
use tabby_common::config::RepositoryConfig;
use tabby_db::DbConn;
use tabby_inference::Embedding;
use tabby_schema::{
    integration::IntegrationService,
    repository::{GitRepositoryService, ThirdPartyRepositoryService},
    web_crawler::WebCrawlerService,
};
use third_party_integration::SchedulerGithubGitlabJob;
use tracing::warn;
use web_crawler::WebCrawlerJob;

use self::{db::DbMaintainanceJob, third_party_integration::SyncIntegrationJob};

#[derive(PartialEq, Debug, Serialize, Deserialize)]
pub enum BackgroundJobEvent {
    SchedulerGitRepository(RepositoryConfig),
    SchedulerGithubGitlabRepository(ID),
    SyncThirdPartyRepositories(ID),
    WebCrawler(String),
}

impl BackgroundJobEvent {
    pub fn to_command(&self) -> String {
        serde_json::to_string(self).expect("Failed to serialize background job event")
    }
}

pub async fn start(
    db: DbConn,
    git_repository_service: Arc<dyn GitRepositoryService>,
    third_party_repository_service: Arc<dyn ThirdPartyRepositoryService>,
    integration_service: Arc<dyn IntegrationService>,
    _web_crawler_service: Arc<dyn WebCrawlerService>,
    embedding: Arc<dyn Embedding>,
    sender: tokio::sync::mpsc::UnboundedSender<BackgroundJobEvent>,
) {
    let mut hourly =
        CronStream::new(Schedule::from_str("@hourly").expect("Invalid cron expression"))
            .into_stream();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                job = db.get_next_job_to_execute() => {
                    let Some(job) = job else {
                        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                        continue;
                    };

                    let event = serde_json::from_str::<BackgroundJobEvent>(&job.command).expect("Failed to deserialize background job event");

                    match event {
                        BackgroundJobEvent::SchedulerGitRepository(repository_config) => {
                            let mut job_logger = JobLogger::new(db.clone(), job.id).await;
                            let job = SchedulerGitJob::new(repository_config);
                            if let Err(err) = job.run(job_logger.clone(), embedding.clone()).await {
                                cprintln!(job_logger, "{:?}", err);
                                job_logger.complete(-1).await;
                            } else {
                                job_logger.complete(0).await;
                            }
                        },
                        BackgroundJobEvent::SyncThirdPartyRepositories(integration_id) => {
                            let mut job_logger = JobLogger::new(db.clone(), job.id).await;
                            let job = SyncIntegrationJob::new(integration_id);
                            if let Err(err) = job.run(third_party_repository_service.clone()).await {
                                warn!("Sync integration job failed: {err:?}");
                                job_logger.complete(-1).await;
                            } else {
                                job_logger.complete(0).await;
                            }
                        }
                        BackgroundJobEvent::SchedulerGithubGitlabRepository(integration_id) => {
                            let mut job_logger = JobLogger::new(db.clone(), job.id).await;
                            let job = SchedulerGithubGitlabJob::new(integration_id);
                            if let Err(err) = job.run(job_logger.clone(), embedding.clone(), third_party_repository_service.clone(), integration_service.clone()).await {
                                cprintln!(job_logger, "{:?}", err);
                                job_logger.complete(-1).await;
                            } else {
                                job_logger.complete(0).await;
                            }
                        }
                        BackgroundJobEvent::WebCrawler(url) => {
                            let mut job_logger = JobLogger::new(db.clone(), job.id).await;
                            let job = WebCrawlerJob::new(url);

                            if let Err(err) = job.run(job_logger.clone(), embedding.clone()).await {
                                cprintln!(job_logger, "Web indexing process failed: do you have Katana installed? (https://github.com/projectdiscovery/katana) \n{err:?}");
                                job_logger.complete(-1).await;
                            } else {
                                job_logger.complete(0).await;
                            }
                        }
                    }
                },
                Some(now) = hourly.next() => {
                    if let Err(err) = DbMaintainanceJob::cron(now, db.clone()).await {
                        warn!("Database maintainance failed: {:?}", err);
                    }

                    if let Err(err) = SchedulerGitJob::cron(now, git_repository_service.clone(), sender.clone()).await {
                        warn!("Scheduler job failed: {:?}", err);
                    }

                    if let Err(err) = SyncIntegrationJob::cron(now, sender.clone(), integration_service.clone()).await {
                        warn!("Sync integration job failed: {:?}", err);
                    }

                    if let Err(err) = SchedulerGithubGitlabJob::cron(now, sender.clone(), third_party_repository_service.clone()).await {
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
