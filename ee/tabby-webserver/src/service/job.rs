use anyhow::Result;
use async_trait::async_trait;

use super::db::DbConn;
use crate::schema::job::{JobRun, JobService};

#[async_trait]
impl JobService for DbConn {
    async fn list_job_runs(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<JobRun>> {
        let runs = match (first, last) {
            (Some(first), None) => {
                let after = after.map(|x| x.parse::<i32>()).transpose()?;
                self.list_job_runs_with_filter(Some(first), after, false)
                    .await?
            }
            (None, Some(last)) => {
                let before = before.map(|x| x.parse::<i32>()).transpose()?;
                self.list_job_runs_with_filter(Some(last), before, true)
                    .await?
            }
            _ => self.list_job_runs_with_filter(None, None, false).await?,
        };

        Ok(runs.into_iter().map(|x| x.into()).collect())
    }
}
