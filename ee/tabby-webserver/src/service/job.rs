use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    job::{JobInfo, JobRun, JobService, JobStats},
    AsRowid, Result,
};
use tracing::warn;

use super::graphql_pagination_to_filter;
use crate::service::background_job::BackgroundJobEvent;

struct JobControllerImpl {
    db: DbConn,
    background_job_sender: tokio::sync::mpsc::UnboundedSender<BackgroundJobEvent>,
}

pub async fn create(
    db: DbConn,
    background_job_sender: tokio::sync::mpsc::UnboundedSender<BackgroundJobEvent>,
) -> impl JobService {
    JobControllerImpl {
        db,
        background_job_sender,
    }
}

#[async_trait]
impl JobService for JobControllerImpl {
    async fn trigger(&self, command: String) {
        if let Err(err) = self.db.create_job_run("triggered".into(), command).await {
            warn!("Failed to create job: {:?}", err);
        }
    }

    async fn list(
        &self,
        ids: Option<Vec<ID>>,
        jobs: Option<Vec<String>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<JobRun>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let rowids = ids.map(|ids| {
            ids.into_iter()
                .filter_map(|x| x.as_rowid().ok().map(|x| x as i32))
                .collect()
        });
        Ok(self
            .db
            .list_job_runs_with_filter(rowids, jobs, limit, skip_id, backwards)
            .await?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    async fn get_job_info(&self, command: String) -> Result<JobInfo> {
        let job_run = self.db.get_latest_job_run(command.clone()).await?;
        Ok(JobInfo {
            last_job_run: job_run.map(JobRun::from),
            command,
        })
    }

    async fn compute_stats(&self, jobs: Option<Vec<String>>) -> Result<JobStats> {
        let stats = self.db.compute_job_stats(jobs).await?;
        Ok(JobStats {
            success: stats.success,
            failed: stats.failed,
            pending: stats.pending,
        })
    }
}
