use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLInputObject, GraphQLObject, ID};
use serde::{Deserialize, Serialize};
use tabby_db::slack_workspaces::SlackWorkspaceDAO;
use validator::Validate;

use super::Context;
use crate::{job::JobInfo, juniper::relay, Result};

#[derive(Serialize, Deserialize, Validate, GraphQLInputObject)]
pub struct CreateSlackWorkspaceIntegrationInput {
    #[validate(length(min = 1, max = 100))]
    pub workspace_name: String,
    pub workspace_id: String,
    pub bot_token: String,
    pub channels: Option<Vec<String>>,
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct SlackWorkspace {
    pub id: ID,
    pub bot_token: String,
    pub workspace_name: String,
    //if no channels are provided, it will post to all channels
    pub channels: Option<Vec<String>>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub job_info: JobInfo,
}

impl relay::NodeType for SlackWorkspace {
    type Cursor = String;
    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }
    fn connection_type_name() -> &'static str {
        "SlackWorkspaceConnection"
    }
    fn edge_type_name() -> &'static str {
        "SlackWorkspaceEdge"
    }
}

impl SlackWorkspace {
    pub fn source_id(&self) -> String {
        Self::format_source_id(&self.id)
    }
    pub fn format_source_id(id: &ID) -> String {
        format!("slack_workspace:{}", id)
    }
}

#[async_trait]
pub trait SlackWorkspaceIntegrationService: Send + Sync {
    async fn list(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<SlackWorkspace>>;

    async fn create(&self, input: CreateSlackWorkspaceIntegrationInput) -> Result<ID>;

    async fn delete(&self, id: ID) -> Result<bool>;

    //TODO: test code, remove later
    // async fn trigger_slack_integration_job(&self, id: ID) -> Result<JobInfo>;
}

pub fn to_slack_workspace(dao: SlackWorkspaceDAO, job_info: JobInfo) -> SlackWorkspace {
    SlackWorkspace {
        id: ID::from(dao.id.to_string()),
        bot_token: dao.bot_token,
        workspace_name: dao.workspace_name,
        created_at: dao.created_at,
        updated_at: dao.updated_at,
        job_info: JobInfo {
            last_job_run: job_info.last_job_run,
            command: job_info.command,
        },
        channels: Some(dao.get_channels().unwrap_or_default()),
    }
}
