use anyhow::Context;
use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    job::{JobInfo, JobRun, JobService, JobStats},
    AsID, AsRowid, Result,
};

use super::graphql_pagination_to_filter;
use crate::service::background_job::BackgroundJobEvent;

struct JobControllerImpl {
    db: DbConn,
}

pub async fn create(db: DbConn) -> impl JobService {
    JobControllerImpl { db }
}

#[async_trait]
impl JobService for JobControllerImpl {
    async fn trigger(&self, command: String) -> Result<ID> {
        if let Some(job) = self.db.get_latest_job_run(command.clone()).await {
            if job.is_pending() {
                // When there's pending job, return the existing job id
                return Ok(job.id.as_id());
            }
        };

        let event = serde_json::from_str::<BackgroundJobEvent>(&command)
            .context("Failed to parse background job event")?;
        Ok(self
            .db
            .create_job_run(event.name().to_owned(), command)
            .await?
            .as_id())
    }

    async fn clear(&self, command: String) -> Result<usize> {
        let num_deleted = self.db.delete_pending_job_run(&command).await?;
        Ok(num_deleted)
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
        let job_run = self.db.get_latest_job_run(command.clone()).await;
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

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;

    use super::*;
    use crate::background_job::{BackgroundJobEvent, WebCrawlerJob};
    use assert_matches::assert_matches;

    #[tokio::test]
    async fn test_clear() {
        let db = DbConn::new_in_memory().await.unwrap();
        let svc = super::create(db.clone()).await;

        let job1 = BackgroundJobEvent::WebCrawler(WebCrawlerJob::new(
            "s1".into(),
            "http://abc.com".into(),
            None,
        ));

        let job2 = BackgroundJobEvent::WebCrawler(WebCrawlerJob::new(
            "s2".into(),
            "http://def.com".into(),
            None,
        ));

        svc.trigger(job1.to_command()).await.unwrap();

        // job1 is marked as stale.
        let _ = db.finalize_stale_job_runs().await;

        svc.trigger(job2.to_command()).await.unwrap();

        assert_eq!(
            db.get_next_job_to_execute().await.unwrap().command,
            job2.to_command()
        );

        // As job1 is marked as stale, no jobs will be cleared.
        assert_eq!(0, svc.clear(job1.to_command()).await.unwrap());
        assert_eq!(1, svc.clear(job2.to_command()).await.unwrap());

        // Regression test case, cleared job shouldn't be pending.
        // job2 started_at is NULL, but exit code is -1
        let job2dao = db.get_latest_job_run(job2.to_command()).await.unwrap();
        assert!(job2dao.started_at.is_none());
        assert_matches!(job2dao.exit_code, Some(-1));
        assert!(!job2dao.is_pending())
    }
}
