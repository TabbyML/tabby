mod db;
mod git;
mod helper;
mod third_party_integration;
mod web_crawler;

use std::{str::FromStr, sync::Arc};

use cron::Schedule;
use futures::StreamExt;
use git::SchedulerGitJob;
use helper::{CronStream, Job, JobLoggerGuard};
use juniper::ID;
use serde::{Deserialize, Serialize};
use tabby_common::config::RepositoryConfig;
use tabby_db::DbConn;
use tabby_inference::Embedding;
use tabby_schema::{
    integration::IntegrationService,
    job::JobService,
    repository::{GitRepositoryService, ThirdPartyRepositoryService},
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
    pub fn name(&self) -> &'static str {
        match self {
            BackgroundJobEvent::SchedulerGitRepository(_) => SchedulerGitJob::NAME,
            BackgroundJobEvent::SchedulerGithubGitlabRepository(_) => {
                SchedulerGithubGitlabJob::NAME
            }
            BackgroundJobEvent::SyncThirdPartyRepositories(_) => SyncIntegrationJob::NAME,
            BackgroundJobEvent::WebCrawler(_) => WebCrawlerJob::NAME,
        }
    }

    pub fn to_command(&self) -> String {
        serde_json::to_string(self).expect("Failed to serialize background job event")
    }
}

pub async fn start(
    db: DbConn,
    job_service: Arc<dyn JobService>,
    git_repository_service: Arc<dyn GitRepositoryService>,
    third_party_repository_service: Arc<dyn ThirdPartyRepositoryService>,
    integration_service: Arc<dyn IntegrationService>,
    embedding: Arc<dyn Embedding>,
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

                    let _ = JobLoggerGuard::new(db.clone(), job.id);
                    let Ok(event) = serde_json::from_str::<BackgroundJobEvent>(&job.command) else {
                        log::info!("Failed to parse background job event, marking it as failed");
                        continue;
                    };

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
                        BackgroundJobEvent::WebCrawler(url) => {
                            let job = WebCrawlerJob::new(url);
                            job.run(embedding.clone()).await
                        }
                    } {
                        log::error!("{:?}", err);
                    }
                },
                Some(now) = hourly.next() => {
                    if let Err(err) = DbMaintainanceJob::cron(now, db.clone()).await {
                        warn!("Database maintainance failed: {:?}", err);
                    }

                    if let Err(err) = SchedulerGitJob::cron(now, git_repository_service.clone(), job_service.clone()).await {
                        warn!("Scheduler job failed: {:?}", err);
                    }

                    if let Err(err) = SyncIntegrationJob::cron(now, integration_service.clone(), job_service.clone()).await {
                        warn!("Sync integration job failed: {:?}", err);
                    }

                    if let Err(err) = SchedulerGithubGitlabJob::cron(now, third_party_repository_service.clone(), job_service.clone()).await {
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
