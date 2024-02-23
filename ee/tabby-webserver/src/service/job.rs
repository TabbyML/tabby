use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;

use super::{graphql_pagination_to_filter, AsID, AsRowid};
use crate::schema::job::{JobRun, JobService};
use crate::schema::Result;

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
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<JobRun>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let rowids = ids.map(|ids| ids.into_iter().filter_map(|x| x.as_rowid().ok()).collect());
        Ok(self
            .list_job_runs_with_filter(rowids, limit, skip_id, backwards)
            .await?
            .into_iter()
            .map(Into::into)
            .collect())
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
            .list_job_runs(None, None, None, None, None)
            .await
            .unwrap();
        let job = job.first().unwrap();
        assert_eq!(job.job, "test-job");
        assert_eq!(job.stdout, "stdout");
        assert_eq!(job.stderr, "stderr");
        assert_eq!(job.exit_code, Some(0));

        let job = svc
            .list_job_runs(Some(vec![id]), None, None, None, None)
            .await
            .unwrap();
        assert_eq!(job.len(), 1)
    }
}
