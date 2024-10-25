use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{prelude::FromRow, query, query_as};
use tabby_db_macros::query_paged_as;

use crate::DbConn;

#[derive(Debug, FromRow, Serialize, Deserialize, Clone)]
pub struct SlackWorkspaceDAO {
    pub id: i64,
    pub workspace_name: String,
    pub workspace_id: String,
    pub bot_token: String,
    pub channels: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl SlackWorkspaceDAO {
    pub fn get_channels(&self) -> Result<Vec<String>, serde_json::Error> {
        serde_json::from_value(self.channels.clone())
    }
}

impl DbConn {
    pub async fn list_slack_workspace_integrations(
        &self,
        ids: Option<Vec<i64>>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<SlackWorkspaceDAO>> {
        let mut conditions = vec![];
        if let Some(ids) = ids {
            let ids: Vec<String> = ids.iter().map(i64::to_string).collect();
            let ids = ids.join(", ");
            conditions.push(format!("id in ({ids})"));
        }
        let condition = (!conditions.is_empty()).then_some(conditions.join(" AND "));
        let integrations = query_paged_as!(
            SlackWorkspaceDAO,
            "slack_workspaces",
            [
                "id",
                "workspace_name",
                "workspace_id",
                "bot_token",
                "channels",
                "created_at" as "created_at: DateTime<Utc>",
                "updated_at" as "updated_at: DateTime<Utc>"
            ],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(integrations)
    }

    pub async fn create_slack_workspace_integration(
        &self,
        workspace_name: String,
        workspace_id: String,
        bot_token: String,
        channels: Option<Vec<String>>,
    ) -> Result<i64> {
        let channels_json = serde_json::to_value(channels.unwrap_or_default())?;

        let res = query!(
            "INSERT INTO slack_workspaces(workspace_name, workspace_id, bot_token, channels) VALUES (?, ?, ?, ?);",
            workspace_name,
            workspace_id,
            bot_token,
            channels_json
        )
        .execute(&self.pool)
        .await?;

        Ok(res.last_insert_rowid())
    }

    pub async fn delete_slack_workspace_integration(&self, id: i64) -> Result<bool> {
        query!("DELETE FROM slack_workspaces WHERE id = ?;", id)
            .execute(&self.pool)
            .await?;
        Ok(true)
    }
    pub async fn get_slack_workspace_integration(
        &self,
        id: i64,
    ) -> Result<Option<SlackWorkspaceDAO>> {
        let integration = query_as!(
            SlackWorkspaceDAO,
            r#"SELECT 
                id, 
                workspace_name, 
                workspace_id, 
                bot_token, 
                channels as "channels: Value",
                created_at as "created_at!: DateTime<Utc>",
                updated_at as "updated_at!: DateTime<Utc>"
            FROM slack_workspaces 
            WHERE id = ?"#,
            id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(integration)
    }

    pub async fn update_slack_workspace_integration(
        &self,
        id: i64,
        workspace_name: String,
        workspace_id: String,
        bot_token: String,
        channels: Option<Vec<String>>,
    ) -> Result<()> {
        let channels_json = serde_json::to_value(channels.unwrap_or_default())?;
        let rows = query!(
            "UPDATE slack_workspaces 
             SET workspace_name = ?, workspace_id = ?, bot_token = ?, channels = ? 
             WHERE id = ?",
            workspace_name,
            workspace_id,
            bot_token,
            channels_json,
            id
        )
        .execute(&self.pool)
        .await?;

        if rows.rows_affected() == 1 {
            Ok(())
        } else {
            Err(anyhow!("failed to update: slack workspace not found"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DbConn;

    #[tokio::test]
    async fn test_update_slack_workspace() {
        let conn = DbConn::new_in_memory().await.unwrap();

        let channels = Some(vec!["channel1".to_string(), "channel2".to_string()]);
        let id = conn
            .create_slack_workspace_integration(
                "test_workspace".into(),
                "W123456".into(),
                "xoxb-test-token".into(),
                channels,
            )
            .await
            .unwrap();

        let workspace = conn
            .get_slack_workspace_integration(id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(workspace.workspace_name, "test_workspace");
        assert_eq!(workspace.workspace_id, "W123456");

        let new_channels = Some(vec!["new_channel1".to_string(), "new_channel2".to_string()]);
        conn.update_slack_workspace_integration(
            id,
            "updated_workspace".into(),
            "W789012".into(),
            "xoxb-new-token".into(),
            new_channels,
        )
        .await
        .unwrap();

        let updated_workspace = conn
            .get_slack_workspace_integration(id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated_workspace.workspace_name, "updated_workspace");
        assert_eq!(updated_workspace.workspace_id, "W789012");
        assert_eq!(updated_workspace.bot_token, "xoxb-new-token");
    }
}
