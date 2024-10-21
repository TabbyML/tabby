use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    job::{JobInfo, JobService},
    slack_workspaces::{
        to_slack_workspace_integration, CreateSlackWorkspaceIntegrationInput,
        SlackWorkspaceIntegration, SlackWorkspaceIntegrationService,
    },
    AsID, AsRowid, CoreError, Result,
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
                integration.get_channels(),
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
        //create in db
        let id = self
            .db
            .create_slack_workspace_integration(
                input.workspace_name,
                input.workspace_id,
                input.bot_token,
                input.channels,
            )
            .await?;

        //trigger in background job
        let _ = self
            .job_service
            .trigger(
                BackgroundJobEvent::SlackIntegration(SlackIntegrationJob::new(
                    id.to_string(),
                    input.workspace_id,
                    input.bot_token,
                    input.channels,
                ))
                .to_command(),
            )
            .await;

        Ok(id.as_id())
    }

    async fn delete_slack_workspace_integration(&self, id: ID) -> Result<bool> {
        let rowid = id.as_rowid()?;
        let integration = {
            let mut x = self
                .db
                .list_slack_workspace_integrations(Some(vec![rowid]), None, None, false)
                .await?;

            x.pop()
                .context("Slack workspace integration doesn't exist")?
        };
        self.db.delete_slack_workspace_integration(rowid).await?;
        self.job_service
            .clear(
                BackgroundJobEvent::SlackIntegration(SlackIntegrationJob::new(
                    rowid.to_string(),
                    integration.workspace_id,
                    integration.bot_token,
                    integration.get_channels(),
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
    use super::*;
    use tabby_db::DbConn;

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
            channels: vec![],
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

        // Test trigger job
        let job_info = service
            .trigger_slack_integration_job(id.clone())
            .await
            .unwrap();
        assert!(job_info.last_job_run.is_some());

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
