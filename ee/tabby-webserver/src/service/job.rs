use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;

use super::{graphql_pagination_to_filter, AsID, AsRowid};
use crate::schema::{
    job::{JobRun, JobService, JobStats},
    Result,
};

#[async_trait]
impl JobService for DbConn {
    async fn create_job_run(&self, name: String) -> Result<ID> {
        Ok(self.create_job_run(name).await.map(|x| x.as_id())?)
    }

    async fn update_job_stdout(&self, id: &ID, stdout: String) -> Result<()> {
        self.update_job_stdout(id.as_rowid()?, stdout).await?;
        Ok(())
    }

    async fn update_job_stderr(&self, id: &ID, stderr: String) -> Result<()> {
        self.update_job_stderr(id.as_rowid()?, stderr).await?;
        Ok(())
    }

    async fn complete_job_run(&self, id: &ID, exit_code: i32) -> Result<()> {
        self.update_job_status(id.as_rowid()?, exit_code).await?;
        Ok(())
    }

    async fn cleanup_stale_job_runs(&self) -> Result<()> {
        (self as &DbConn).cleanup_stale_job_runs().await?;
        Ok(())
    }

    async fn list_job_runs(
        &self,
        ids: Option<Vec<ID>>,
        jobs: Option<Vec<String>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<JobRun>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let rowids = ids.map(|ids| ids.into_iter().filter_map(|x| x.as_rowid().ok()).collect());
        Ok(self
            .list_job_runs_with_filter(rowids, jobs, limit, skip_id, backwards)
            .await?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    async fn compute_job_run_stats(&self, jobs: Option<Vec<String>>) -> Result<JobStats> {
        let stats = (self as &DbConn).compute_job_stats(jobs).await?;
        Ok(JobStats {
            success: stats.success,
            failed: stats.failed,
            pending: stats.pending,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_job_service() {
        let svc: Box<dyn JobService> = Box::new(DbConn::new_in_memory().await.unwrap());

        let id = svc.create_job_run("test-job".to_owned()).await.unwrap();
        svc.update_job_stdout(&id, "stdout".to_owned())
            .await
            .unwrap();
        svc.update_job_stderr(&id, "stderr".to_owned())
            .await
            .unwrap();
        svc.complete_job_run(&id, 0).await.unwrap();

        let job = svc
            .list_job_runs(None, None, None, None, None, None)
            .await
            .unwrap();
        let job = job.first().unwrap();
        assert_eq!(job.job, "test-job");
        assert_eq!(job.stdout, "stdout");
        assert_eq!(job.stderr, "stderr");
        assert_eq!(job.exit_code, Some(0));

        let job = svc
            .list_job_runs(Some(vec![id]), None, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(job.len(), 1)
    }

    #[tokio::test]
    async fn test_job_stats() {
        let db = DbConn::new_in_memory().await.unwrap();
        let jobs: Box<dyn JobService> = Box::new(db);

        let id = jobs.create_job_run("test-job".into()).await.unwrap();
        jobs.complete_job_run(&id, 0).await.unwrap();

        let id2 = jobs.create_job_run("test-job".into()).await.unwrap();
        jobs.complete_job_run(&id2, 1).await.unwrap();

        jobs.create_job_run("pending-job".into()).await.unwrap();

        let stats = jobs.compute_job_run_stats(None).await.unwrap();

        assert_eq!(stats.success, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.pending, 1);

        let stats = jobs
            .compute_job_run_stats(Some(vec!["test-job".into()]))
            .await
            .unwrap();

        assert_eq!(stats.success, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.pending, 0);
    }
}
