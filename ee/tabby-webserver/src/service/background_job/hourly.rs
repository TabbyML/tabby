use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;
use tabby_schema::{
    context::ContextService,
    integration::IntegrationService,
    job::JobService,
    repository::{GitRepositoryService, RepositoryService, ThirdPartyRepositoryService},
};

use super::helper::Job;
use crate::service::background_job::{
    db::DbMaintainanceJob, IndexGarbageCollection, SchedulerGitJob, SchedulerGithubGitlabJob,
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
        integration_service: Arc<dyn IntegrationService>,
        third_party_repository_service: Arc<dyn ThirdPartyRepositoryService>,
        repository_service: Arc<dyn RepositoryService>,
    ) -> tabby_schema::Result<()> {
        let now = Utc::now();

        if let Err(err) = DbMaintainanceJob::cron(now, context_service.clone(), db.clone()).await {
            logkit::warn!("Database maintainance failed: {:?}", err);
        }

        if let Err(err) =
            SchedulerGitJob::cron(now, git_repository_service.clone(), job_service.clone()).await
        {
            logkit::warn!("Scheduler job failed: {:?}", err);
        }

        if let Err(err) =
            SyncIntegrationJob::cron(now, integration_service.clone(), job_service.clone()).await
        {
            logkit::warn!("Sync integration job failed: {:?}", err);
        }

        if let Err(err) = SchedulerGithubGitlabJob::cron(
            now,
            third_party_repository_service.clone(),
            job_service.clone(),
        )
        .await
        {
            logkit::warn!("Index issues job failed: {err:?}");
        }

        if let Err(err) = IndexGarbageCollection
            .run(repository_service.clone(), context_service.clone())
            .await
        {
            logkit::warn!("Index garbage collection job failed: {err:?}");
        }
        Ok(())
    }
}
