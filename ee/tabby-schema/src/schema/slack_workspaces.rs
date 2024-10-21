use super::{job, Context};
use crate::{job::JobInfo, juniper::relay, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{graphql_object, GraphQLInputObject, GraphQLObject, ID};
use serde::{Deserialize, Serialize};
use tabby_db::slack_workspaces::SlackWorkspaceIntegrationDAO;
use validator::Validate;

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct SlackWorkspaceIntegration {
    pub id: ID,
    pub workspace_id: String,
    pub bot_token: String,
    pub workspace_name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub job_info: JobInfo,
}

impl relay::NodeType for SlackWorkspaceIntegration {
    type Cursor = String;
    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }
    fn connection_type_name() -> &'static str {
        "SlackWorkspaceIntegrationConnection"
    }
    fn edge_type_name() -> &'static str {
        "SlackWorkspaceIntegrationEdge"
    }
}

impl SlackWorkspaceIntegration {
    pub fn source_id(&self) -> String {
        Self::format_source_id(&self.id)
    }
    pub fn format_source_id(id: &ID) -> String {
        format!("slack_workspace:{}", id)
    }
}

#[derive(Serialize, Deserialize, Validate, GraphQLInputObject)]
pub struct CreateSlackWorkspaceIntegrationInput {
    #[validate(length(min = 1, max = 100))]
    pub workspace_name: String,
    pub workspace_id: String,
    pub bot_token: String,
    pub channels: Option<Vec<String>>,
}

// #[derive(Serialize, Deserialize, GraphQLInputObject)]
// pub struct SlackChannelInput {
//     pub id: String,
//     pub name: String,
// }

#[async_trait]
pub trait SlackWorkspaceIntegrationService: Send + Sync {
    async fn list_slack_workspace_integrations(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<SlackWorkspaceIntegration>>;

    async fn create_slack_workspace_integration(
        &self,
        input: CreateSlackWorkspaceIntegrationInput,
    ) -> Result<ID>;

    async fn delete_slack_workspace_integration(&self, id: ID) -> Result<bool>;

    //TODO: test code, remove later
    // async fn trigger_slack_integration_job(&self, id: ID) -> Result<JobInfo>;
}

pub fn to_slack_workspace_integration(
    dao: SlackWorkspaceIntegrationDAO,
    job_info: JobInfo,
) -> SlackWorkspaceIntegration {
    SlackWorkspaceIntegration {
        id: ID::from(dao.id.to_string()),
        workspace_id: dao.workspace_id,
        bot_token: dao.bot_token,
        workspace_name: dao.workspace_name,
        created_at: dao.created_at,
        updated_at: dao.updated_at,
        job_info: JobInfo {
            last_job_run: job_info.last_job_run,
            command: job_info.command,
        },
    }
}
