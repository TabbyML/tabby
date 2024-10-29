use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLInputObject, GraphQLObject, ID};
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::Context;
use crate::{job::JobInfo, juniper::relay, Result};

#[derive(Serialize, Deserialize, Validate, GraphQLInputObject)]
pub struct CreateSlackWorkspaceInput {
    #[validate(length(min = 1, max = 100))]
    pub workspace_name: String,
    pub bot_token: String,
    pub channel_ids: Option<Vec<String>>,
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

#[derive(Debug, Clone, GraphQLObject)]
pub struct SlackChannel {
    pub id: String,
    pub name: String,
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
pub trait SlackWorkspaceService: Send + Sync {
    async fn list(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<SlackWorkspace>>;

    async fn create(
        &self,
        workspace_name: String,
        bot_token: String,
        channel_ids: Option<Vec<String>>,
    ) -> Result<ID>;

    async fn delete(&self, id: ID) -> Result<bool>;

    /// List all workspaces
    async fn list_workspaces(&self) -> Result<Vec<SlackWorkspace>>;

    async fn list_visible_channels(&self, bot_token: String) -> Result<Vec<SlackChannel>>;
}
