use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    job::JobService,
    slack_workspaces::{
        to_slack_workspace_integration, CreateSlackWorkspaceIntegrationInput,
        SlackWorkspaceIntegration, SlackWorkspaceIntegrationService,
    },
    AsID, AsRowid, Result,
};

use super::{
    background_job::{slack_integration::SlackIntegrationJob, BackgroundJobEvent},
    graphql_pagination_to_filter,
};

pub fn create(
    db: DbConn,
    job_service: Arc<dyn JobService>,
) -> impl SlackWorkspaceIntegrationService {
    SlackWorkspaceIntegrationServiceImpl { db, job_service }
}

struct SlackWorkspaceIntegrationServiceImpl {
    db: DbConn,
    job_service: Arc<dyn JobService>,
}

#[async_trait]
impl SlackWorkspaceIntegrationService for SlackWorkspaceIntegrationServiceImpl {
    async fn list_slack_workspace_integrations(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<SlackWorkspaceIntegration>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;
        let ids = ids.map(|x| {
            x.iter()
                .filter_map(|x| x.as_rowid().ok())
                .collect::<Vec<_>>()
        });
        let integrations = self
            .db
            .list_slack_workspace_integrations(ids, limit, skip_id, backwards)
            .await?;

        let mut converted_integrations = vec![];

        for integration in integrations {
            let event = BackgroundJobEvent::SlackIntegration(SlackIntegrationJob::new(
                integration.id.to_string(),
                integration.workspace_id.clone(),
                integration.bot_token.clone(),
                Some(integration.get_channels().unwrap_or_default()),
            ));

            let job_info = self.job_service.get_job_info(event.to_command()).await?;
            converted_integrations.push(to_slack_workspace_integration(integration, job_info));
        }
        Ok(converted_integrations)
    }

    async fn create_slack_workspace_integration(
        &self,
        input: CreateSlackWorkspaceIntegrationInput,
    ) -> Result<ID> {
        let workspace_id = input.workspace_id.clone();
        let bot_token = input.bot_token.clone();
        let channels = input.channels.clone();
        //create in db
        let id = self
            .db
            .create_slack_workspace_integration(
                input.workspace_name,
                workspace_id,
                bot_token,
                channels,
            )
            .await?;
        let workspace_id = input.workspace_id.clone();
        let bot_token = input.bot_token.clone();
        let channels = input.channels.clone();
        //trigger in background job
        let _ = self
            .job_service
            .trigger(
                BackgroundJobEvent::SlackIntegration(SlackIntegrationJob::new(
                    id.to_string(),
                    workspace_id,
                    bot_token,
                    channels,
                ))
                .to_command(),
            )
            .await;

        Ok(id.as_id())
    }
    async fn delete_slack_workspace_integration(&self, id: ID) -> Result<bool> {
        let row_id = id.as_rowid()?;

        let integration = {
            let mut x = self
                .db
                .list_slack_workspace_integrations(Some(vec![row_id]), None, None, false)
                .await?;
            x.pop()
                .context("Slack workspace integration doesn't exist")?
        };

        self.db.delete_slack_workspace_integration(row_id).await?;

        // Clone the necessary fields
        let workspace_id = integration.workspace_id.clone();
        let bot_token = integration.bot_token.clone();
        let channels = integration.get_channels().unwrap_or_default();

        self.job_service
            .clear(
                BackgroundJobEvent::SlackIntegration(SlackIntegrationJob::new(
                    row_id.to_string(),
                    workspace_id,
                    bot_token,
                    Some(channels),
                ))
                .to_command(),
            )
            .await?;

        self.job_service
            .trigger(BackgroundJobEvent::IndexGarbageCollection.to_command())
            .await?;

        Ok(true)
    }
    // async fn trigger_slack_integration_job(&self, id: ID) -> Result<JobInfo> {
    //     let integration = self.db.get_slack_workspace_integration(id).await;

    //     let event = BackgroundJobEvent::SlackIntegration(SlackIntegrationJob::new(id,));
    //     let job_id = self.job_service.trigger(event.to_command()).await?;
    //     self.job_service.get_job_info(job_id).await
    // }
}

#[cfg(test)]
mod tests {
    use tabby_db::DbConn;

    use super::*;

    #[tokio::test]
    async fn test_slack_workspace_integration_service() {
        let db = DbConn::new_in_memory().await.unwrap();
        let job = Arc::new(crate::service::job::create(db.clone()).await);
        let service = create(db.clone(), job.clone());

        // Test create
        let input = CreateSlackWorkspaceIntegrationInput {
            workspace_name: "Test Workspace".to_string(),
            workspace_id: "W12345".to_string(),
            bot_token: "xoxb-test-token".to_string(),
            channels: Some(vec![]),
        };
        let id = service
            .create_slack_workspace_integration(input)
            .await
            .unwrap();

        // Test list
        let integrations = service
            .list_slack_workspace_integrations(None, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(1, integrations.len());
        assert_eq!(id, integrations[0].id);

        // Test delete
        let result = service
            .delete_slack_workspace_integration(id)
            .await
            .unwrap();
        assert!(result);

        let integrations = service
            .list_slack_workspace_integrations(None, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(0, integrations.len());
    }
}
