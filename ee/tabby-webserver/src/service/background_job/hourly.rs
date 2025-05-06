use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;
use tabby_schema::{
    context::ContextService,
    ingestion::IngestionService,
    integration::IntegrationService,
    job::JobService,
    repository::{GitRepositoryService, RepositoryService, ThirdPartyRepositoryService},
};

use super::helper::Job;
use crate::service::background_job::{
    db::DbMaintenanceJob, IndexGarbageCollection, SchedulerGitJob, SchedulerGithubGitlabJob,
    SyncIntegrationJob,
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HourlyJob;

impl Job for HourlyJob {
    const NAME: &'static str = "hourly";
}

impl HourlyJob {
    pub async fn run(
        &self,
        db: DbConn,
        context_service: Arc<dyn ContextService>,
        git_repository_service: Arc<dyn GitRepositoryService>,
        job_service: Arc<dyn JobService>,
        ingestion_service: Arc<dyn IngestionService>,
        integration_service: Arc<dyn IntegrationService>,
        third_party_repository_service: Arc<dyn ThirdPartyRepositoryService>,
        repository_service: Arc<dyn RepositoryService>,
    ) -> tabby_schema::Result<()> {
        let now = Utc::now();
        let mut has_error = false;

        if let Err(err) = DbMaintenanceJob::cron(now, context_service.clone(), db.clone()).await {
            has_error = true;
            logkit::warn!("Database maintenance failed: {:?}", err);
        }

        if let Err(err) =
            SchedulerGitJob::cron(now, git_repository_service.clone(), job_service.clone()).await
        {
            has_error = true;
            logkit::warn!("Scheduler job failed: {:?}", err);
        }

        if let Err(err) =
            SyncIntegrationJob::cron(now, integration_service.clone(), job_service.clone()).await
        {
            has_error = true;
            logkit::warn!("Sync integration job failed: {:?}", err);
        }

        if let Err(err) = SchedulerGithubGitlabJob::cron(
            now,
            third_party_repository_service.clone(),
            job_service.clone(),
        )
        .await
        {
            has_error = true;
            logkit::warn!("Index issues job failed: {err:?}");
        }

        if let Err(err) = IndexGarbageCollection
            .run(
                repository_service.clone(),
                context_service.clone(),
                ingestion_service.clone(),
            )
            .await
        {
            has_error = true;
            logkit::warn!("Index garbage collection job failed: {err:?}");
        }

        if has_error {
            Err(tabby_schema::CoreError::Other(anyhow::anyhow!(
                "Hourly job failed"
            )))
        } else {
            Ok(())
        }
    }
}
