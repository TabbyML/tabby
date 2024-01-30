use std::fmt::Debug;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::GraphQLObject;
use juniper_axum::relay;

use crate::schema::Context;

#[derive(Debug, GraphQLObject)]
#[graphql(context = Context)]
pub struct JobRun {
    pub id: juniper::ID,
    pub job_name: String,
    pub start_time: DateTime<Utc>,
    pub finish_time: Option<DateTime<Utc>>,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
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
    async fn list_job_runs(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<(Vec<JobRun>, usize)>;
}
