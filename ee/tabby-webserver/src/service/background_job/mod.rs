mod daily;
mod db;
mod git;
mod helper;
mod hourly;
mod index_commits;
mod index_garbage_collection;
mod index_ingestion;
mod index_pages;
mod license_check;
mod third_party_integration;
mod web_crawler;

use std::{env, str::FromStr, sync::Arc};

use cron::Schedule;
use daily::DailyJob;
use futures::StreamExt;
use git::SchedulerGitJob;
use helper::{CronStream, Job, JobLogger};
use hourly::HourlyJob;
use index_garbage_collection::IndexGarbageCollection;
use index_ingestion::SyncIngestionIndexJob;
use index_pages::SyncPageIndexJob;
use juniper::ID;
use license_check::LicenseCheckJob;
use serde::{Deserialize, Serialize};
use tabby_common::config::CodeRepository;
use tabby_db::DbConn;
use tabby_inference::Embedding;
use tabby_schema::{
    context::ContextService,
    ingestion::IngestionService,
    integration::IntegrationService,
    job::JobService,
    license::LicenseService,
    notification::{NotificationRecipient, NotificationService},
    page::PageService,
    repository::{GitRepositoryService, RepositoryService, ThirdPartyRepositoryService},
    AsID,
};
pub use third_party_integration::error::octocrab_error_message;
use third_party_integration::SchedulerGithubGitlabJob;
use tracing::{debug, warn};
use url::Url;
pub use web_crawler::WebCrawlerJob;

use self::third_party_integration::SyncIntegrationJob;

// Sharding configuration constants
pub const REPOSITORIES_PER_SHARD: usize = 7;
pub const SHARDING_THRESHOLD: usize = 20;

/// Calculate the current shard for repository processing
/// Returns Some(shard) if sharding should be used, None otherwise
fn calculate_current_shard(number_of_repo: usize, timestamp_seconds: i64) -> Option<usize> {
    // Only run on TABBY_INDEX_REPO_IN_SHARD is not empty and number_of_repo > SHARDING_THRESHOLD
    // otherwise return None
    if !(env::var("TABBY_INDEX_REPO_IN_SHARD").is_ok_and(|v| !v.is_empty())
        && number_of_repo > SHARDING_THRESHOLD)
    {
        return None;
    }

    // `number_of_repo + REPOSITORIES_PER_SHARD - 1` because we should ceil number_of_repo
    let number_of_shard = number_of_repo.div_ceil(REPOSITORIES_PER_SHARD);
    let timestamp = timestamp_seconds as usize;
    Some((timestamp / 3600) % number_of_shard)
}

/// Check if a repository should be processed based on sharding
fn should_process_repository(
    repo_index: usize,
    current_shard: Option<usize>,
    number_of_repo: usize,
) -> bool {
    let Some(current_shard) = current_shard else {
        return true; // No sharding, process all repositories
    };

    let number_of_shard = number_of_repo.div_ceil(REPOSITORIES_PER_SHARD); // Math.ceil
    repo_index % number_of_shard == current_shard
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum BackgroundJobEvent {
    SchedulerGitRepository(CodeRepository),
    SchedulerGithubGitlabRepository(ID),
    SyncThirdPartyRepositories(ID),
    WebCrawler(WebCrawlerJob),
    IndexGarbageCollection,
    SyncIngestionIndex,
    SyncPagesIndex,
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
            BackgroundJobEvent::SyncPagesIndex => SyncPageIndexJob::NAME,
            BackgroundJobEvent::WebCrawler(_) => WebCrawlerJob::NAME,
            BackgroundJobEvent::IndexGarbageCollection => IndexGarbageCollection::NAME,
            BackgroundJobEvent::SyncIngestionIndex => SyncIngestionIndexJob::NAME,
            BackgroundJobEvent::Hourly => HourlyJob::NAME,
            BackgroundJobEvent::Daily => DailyJob::NAME,
        }
    }

    pub fn to_command(&self) -> String {
        serde_json::to_string(self).expect("Failed to serialize background job event")
    }
}

async fn background_job_notification_name(
    integration_service: Arc<dyn IntegrationService>,
    third_party_repository_service: Arc<dyn ThirdPartyRepositoryService>,
    event: &BackgroundJobEvent,
) -> String {
    match event {
        BackgroundJobEvent::SchedulerGitRepository(repo) => {
            if let Ok(url) = Url::parse(&repo.git_url) {
                format!(
                    "Indexing Git Repository {}",
                    url.path()
                        .strip_suffix(".git")
                        .unwrap_or_else(|| url.path())
                        .rsplit('/')
                        .next()
                        .unwrap_or("")
                )
            } else {
                repo.git_url.clone()
            }
        }
        BackgroundJobEvent::SchedulerGithubGitlabRepository(id) => {
            if let Ok(repo) = third_party_repository_service
                .get_provided_repository(id)
                .await
            {
                format!(
                    "Indexing {} Repository {}",
                    integration_service
                        .get_integration(&repo.integration_id)
                        .await
                        .map(|integration| integration.display_name)
                        .unwrap_or_else(|_| String::new()),
                    repo.display_name
                )
            } else {
                format!("Indexing Third Party Repository {id}")
            }
        }
        BackgroundJobEvent::SyncThirdPartyRepositories(id) => {
            if let Ok(repo) = integration_service.get_integration(id).await {
                format!("Loading {} Repositories", repo.display_name)
            } else {
                format!("Loading Third Party Repositories {id}")
            }
        }
        BackgroundJobEvent::SyncPagesIndex => "Pages Indexing".into(),
        BackgroundJobEvent::WebCrawler(doc) => {
            if let Ok(url) = Url::parse(&doc.url) {
                format!(
                    "Indexing Web {}",
                    url.host_str().unwrap_or_else(|| &doc.url)
                )
            } else {
                format!("Indexing Web {}", doc.url)
            }
        }
        BackgroundJobEvent::SyncIngestionIndex => "Ingestion Indexing".into(),
        BackgroundJobEvent::IndexGarbageCollection => "Index Garbage Collection".into(),
        BackgroundJobEvent::Hourly => "Hourly".into(),
        BackgroundJobEvent::Daily => "Daily".into(),
    }
}

async fn notify_job_error(
    notification_service: Arc<dyn NotificationService>,
    integration_service: Arc<dyn IntegrationService>,
    third_party_repository_service: Arc<dyn ThirdPartyRepositoryService>,
    err: &str,
    event: &BackgroundJobEvent,
    id: i64,
) {
    warn!("job {:?} failed: {:?}", event, err);
    let name = background_job_notification_name(
        integration_service,
        third_party_repository_service,
        event,
    )
    .await;
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
    ingestion_service: Arc<dyn IngestionService>,
    repository_service: Arc<dyn RepositoryService>,
    page_service: Option<Arc<dyn PageService>>,
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

    let mut ten_seconds =
        CronStream::new(Schedule::from_str("*/10 * * * * *").expect("Invalid cron expression"))
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

                    let logger = match JobLogger::new(db.clone(), job.id) {
                        Ok(logger) => logger,
                        Err(err) => {
                            warn!("Failed to create job logger: {:?}", err);
                            continue;
                        }
                    };
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
                        BackgroundJobEvent::SyncPagesIndex => {
                            if let Some(page_service) = page_service.clone() {
                                let job = SyncPageIndexJob;
                                job.run(
                                    page_service.clone(),
                                    embedding.clone(),
                                ).await
                            } else {
                                logkit::info!(exit_code = -1; "No page service available, skipping SyncPagesIndex job");
                                Ok(())
                            }
                        }
                        BackgroundJobEvent::WebCrawler(job) => {
                            job.run(embedding.clone()).await
                        }
                        BackgroundJobEvent::IndexGarbageCollection => {
                            let job = IndexGarbageCollection;
                            job.run(repository_service.clone(), context_service.clone(), ingestion_service.clone()).await
                        }
                        BackgroundJobEvent::SyncIngestionIndex => {
                            let job = SyncIngestionIndexJob;
                            job.run(
                                ingestion_service.clone(),
                                embedding.clone(),
                            ).await
                        }
                        BackgroundJobEvent::Hourly => {
                            let job = HourlyJob;
                            if let Err(e) = job.run(
                                db.clone(),
                                context_service.clone(),
                                git_repository_service.clone(),
                                job_service.clone(),
                                ingestion_service.clone(),
                                integration_service.clone(),
                                third_party_repository_service.clone(),
                                repository_service.clone(),
                            ).await {
                                logkit::warn!("Hourly job failed: {:?}", e);
                            };

                            if let Err(e) = SyncPageIndexJob::cron(job_service.clone()).await {
                                logkit::warn!("Sync page index job failed: {:?}", e);
                            };

                            Ok(())
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
                        notify_job_error(notification_service.clone(), integration_service.clone(), third_party_repository_service.clone(), &err.to_string(), &cloned_event, job.id).await;
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
                Some(_) = ten_seconds.next() => {
                    match SyncIngestionIndexJob::cron(job_service.clone(), ingestion_service.clone()).await {
                        Err(err) => warn!("Schedule ingestion job failed: {}", err),
                        Ok(true) => debug!("Ingestion job scheduled"),
                        Ok(false) => {},
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
