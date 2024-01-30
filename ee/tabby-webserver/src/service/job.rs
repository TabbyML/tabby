use anyhow::Result;
use async_trait::async_trait;
use tabby_db::DbConn;

use super::graphql_pagination_to_filter;
use crate::schema::job::{JobRun, JobService};

#[async_trait]
impl JobService for DbConn {
    async fn list_job_runs(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<(Vec<JobRun>, usize)> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let jobs_count = self.count_rows("job_runs").await?;
        let job_runs = self
            .list_job_runs_with_filter(limit, skip_id, backwards)
            .await?
            .into_iter()
            .map(Into::into)
            .collect();
        Ok((job_runs, jobs_count))
    }
}
