use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use juniper::ID;
use tabby_db::{slack_workspaces::SlackWorkspaceDAO, DbConn};
use tabby_schema::{
    job::{JobInfo, JobService},
    slack_workspaces::{
        CreateSlackWorkspaceInput, SlackChannel, SlackWorkspace, SlackWorkspaceService,
    },
    AsID, AsRowid, Result,
};

use super::{
    background_job::{
        slack::{client::SlackClient, SlackIntegrationJob},
        BackgroundJobEvent,
    },
    graphql_pagination_to_filter,
};

pub fn create(db: DbConn, job_service: Arc<dyn JobService>) -> impl SlackWorkspaceService {
    SlackWorkspaceServiceImpl { db, job_service }
}

struct SlackWorkspaceServiceImpl {
    db: DbConn,
    job_service: Arc<dyn JobService>,
}

#[async_trait]
impl SlackWorkspaceService for SlackWorkspaceServiceImpl {
    async fn list(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<SlackWorkspace>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let ids = ids.map(|x| {
            x.iter()
                .filter_map(|x| x.as_rowid().ok())
                .collect::<Vec<_>>()
        });
        let integrations = self
            .db
            .list_slack_workspace(ids, limit, skip_id, backwards)
            .await?;

        let mut converted_integrations = vec![];

        for integration in integrations {
            let event = BackgroundJobEvent::SlackIntegration(
                SlackIntegrationJob::new(
                    integration.id.to_string(),
                    integration.workspace_name.clone(),
                    integration.bot_token.clone(),
                    Some(integration.get_channels().unwrap_or_default()),
                )
                .await,
            );

            let job_info = self.job_service.get_job_info(event.to_command()).await?;
            converted_integrations.push(to_slack_workspace(integration, job_info));
        }
        Ok(converted_integrations)
    }

    async fn create(&self, input: CreateSlackWorkspaceInput) -> Result<ID> {
        let bot_token = input.bot_token.clone();
        let channels = input.channel_ids.clone();
        //create in db
        let workspace_name = input.workspace_name.clone();

        let id = self
            .db
            .create_slack_workspace(workspace_name.clone(), bot_token, channels)
            .await?;
        let bot_token = input.bot_token.clone();
        let channels = input.channel_ids.clone();
        //trigger in background job
        let _ = self
            .job_service
            .trigger(
                BackgroundJobEvent::SlackIntegration(
                    SlackIntegrationJob::new(id.to_string(), workspace_name, bot_token, channels)
                        .await,
                )
                .to_command(),
            )
            .await;

        Ok(id.as_id())
    }
    async fn delete(&self, id: ID) -> Result<bool> {
        let row_id = id.as_rowid()?;

        let integration = {
            let mut x = self
                .db
                .list_slack_workspace(Some(vec![row_id]), None, None, false)
                .await?;
            x.pop()
                .context("Slack workspace integration doesn't exist")?
        };

        let success = self.db.delete_slack_workspace(row_id).await?;

        if success {
            // Clone the necessary fields
            let workspace_name = integration.workspace_name.clone();
            let bot_token = integration.bot_token.clone();
            let channels = integration.get_channels().unwrap_or_default();

            self.job_service
                .clear(
                    BackgroundJobEvent::SlackIntegration(
                        SlackIntegrationJob::new(
                            row_id.to_string(),
                            workspace_name,
                            bot_token,
                            Some(channels),
                        )
                        .await,
                    )
                    .to_command(),
                )
                .await?;

            self.job_service
                .trigger(BackgroundJobEvent::IndexGarbageCollection.to_command())
                .await?;
        }

        Ok(success)
    }

    async fn list_workspaces(&self) -> Result<Vec<SlackWorkspace>> {
        Ok(self.list(None, None, None, None, None))
    }

    async fn list_visible_channels(bot_token: String) -> Result<Vec<SlackChannel>> {
        let client = SlackClient::new(bot_token.as_str()).await.unwrap();

        Ok(client
            .get_channels()
            .await
            .context("Failed to list slack channels")?)
    }
}

pub fn to_slack_workspace(dao: SlackWorkspaceDAO, job_info: JobInfo) -> SlackWorkspace {
    let channels = dao.clone().get_channels().unwrap_or_default();
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
        channels: Some(channels),
    }
}

#[cfg(test)]
mod tests {

    // #[tokio::test]
    // async fn test_slack_workspace_integration_service() {
    //     let db = DbConn::new_in_memory().await.unwrap();
    //     let job = Arc::new(crate::service::job::create(db.clone()).await);
    //     let service = create(db.clone(), job.clone());

    //     // Test create
    //     let input = CreateSlackWorkspaceIntegrationInput {
    //         workspace_name: "Test Workspace".to_string(),
    //         workspace_id: "W12345".to_string(),
    //         bot_token: "xoxb-test-token".to_string(),
    //         channels: Some(vec![]),
    //     };
    //     let id = service
    //         .create_slack_workspace_integration(input)
    //         .await
    //         .unwrap();

    //     // Test list
    //     let integrations = service
    //         .list_slack_workspace_integrations(None, None, None, None, None)
    //         .await
    //         .unwrap();
    //     assert_eq!(1, integrations.len());
    //     assert_eq!(id, integrations[0].id);

    //     // Test delete
    //     let result = service
    //         .delete_slack_workspace_integration(id)
    //         .await
    //         .unwrap();
    //     assert!(result);

    //     let integrations = service
    //         .list_slack_workspace_integrations(None, None, None, None, None)
    //         .await
    //         .unwrap();
    //     assert_eq!(0, integrations.len());
    // }
}
