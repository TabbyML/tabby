use std::fmt::Debug;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLObject, ID};

use crate::{
    juniper::relay,
    schema::{Context, Result},
};

#[derive(Debug, GraphQLObject)]
#[graphql(context = Context)]
pub struct JobRun {
    pub id: juniper::ID,
    pub job: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub exit_code: Option<i32>,
    pub stdout: String,
}

#[derive(Debug, GraphQLObject)]
#[graphql(context = Context)]
pub struct JobInfo {
    /// Last run of the job.
    pub last_job_run: Option<JobRun>,

    /// The command to submit job run using triggerJobRun mutation.
    pub command: String,
}

#[derive(Debug, GraphQLObject)]
pub struct JobStats {
    pub success: i32,
    pub failed: i32,
    pub pending: i32,
}

impl relay::NodeType for JobRun {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "JobRunConnection"
    }

    fn edge_type_name() -> &'static str {
        "JobRunEdge"
    }
}

#[async_trait]
pub trait JobService: Send + Sync {
    /// Trigger job run.
    async fn trigger(&self, command: String) -> Result<ID>;

    /// Remove pending job run, returns number of jobs being removed.
    async fn clear(&self, command: String) -> Result<usize>;

    async fn get_job_info(&self, command: String) -> Result<JobInfo>;

    async fn list(
        &self,
        ids: Option<Vec<ID>>,
        jobs: Option<Vec<String>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<JobRun>>;

    async fn compute_stats(&self, jobs: Option<Vec<String>>) -> Result<JobStats>;
}
