use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use juniper::ID;
use tabby_db::{slack_workspaces::SlackWorkspaceDAO, DbConn};
use tabby_schema::{
    job::{JobInfo, JobService},
    slack_workspaces::{SlackChannel, SlackWorkspace, SlackWorkspaceService},
    AsID, AsRowid, Result,
};
use tracing::debug;

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

    async fn create(
        &self,
        workspace_name: String,
        bot_token: String,
        channel_ids: Option<Vec<String>>,
    ) -> Result<ID> {
        let bot_token = bot_token.clone();
        let channels = channel_ids.clone();
        //create in db
        let workspace_name = workspace_name.clone();

        let id = self
            .db
            .create_slack_workspace(workspace_name.clone(), bot_token.clone(), channels.clone())
            .await?;
        //trigger in background job
        let _ = self
            .job_service
            .trigger(
                BackgroundJobEvent::SlackIntegration(
                    SlackIntegrationJob::new(
                        id.to_string(),
                        workspace_name,
                        bot_token.clone(),
                        channels,
                    )
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
        Ok(self.list(None, None, None, None, None).await?)
    }

    async fn list_visible_channels(&self, bot_token: String) -> Result<Vec<SlackChannel>> {
        let client = SlackClient::new(bot_token.clone()).await.unwrap();

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

// #[cfg(test)]
// mod tests {

//     use tabby_db::DbConn;

//     use super::*;

//     #[tokio::test]
//     async fn test_duplicate_slack_workspace_error() {
//         let db = DbConn::new_in_memory().await.unwrap();
//         let svc = create(
//             db.clone(),
//             Arc::new(crate::service::job::create(db.clone()).await),
//         );

//         // Create first workspace
//         SlackWorkspaceService::create(
//             &svc,
//             "example".into(),
//             "xoxb-test-token-1".into(),
//             Some(vec!["C1".into()]),
//         )
//         .await
//         .unwrap();

//         // Try to create duplicate workspace
//         let err = SlackWorkspaceService::create(
//             &svc,
//             "example".into(),
//             "xoxb-test-token-1".into(),
//             Some(vec!["C1".into()]),
//         )
//         .await
//         .unwrap_err();

//         assert_eq!(
//             err.to_string(),
//             "A slack workspace with the same name already exists"
//         );
//     }

//     #[tokio::test]
//     async fn test_slack_workspace_mutations() {
//         let db = DbConn::new_in_memory().await.unwrap();
//         let job = Arc::new(crate::service::job::create(db.clone()).await);
//         let service = create(db.clone(), job);

//         // Create first workspace
//         let id_1 = service
//             .create(
//                 "workspace1".into(),
//                 "xoxb-test-token-1".into(),
//                 Some(vec!["C1".into()]),
//             )
//             .await
//             .unwrap();

//         // Create second workspace
//         let id_2 = service
//             .create(
//                 "workspace2".into(),
//                 "xoxb-test-token-2".into(),
//                 Some(vec!["C2".into()]),
//             )
//             .await
//             .unwrap();

//         // Create third workspace
//         service
//             .create(
//                 "workspace3".into(),
//                 "xoxb-test-token-3".into(),
//                 Some(vec!["C3".into()]),
//             )
//             .await
//             .unwrap();

//         // Verify list returns all workspaces
//         assert_eq!(
//             service
//                 .list(None, None, None, None, None)
//                 .await
//                 .unwrap()
//                 .len(),
//             3
//         );

//         // Test delete
//         service.delete(id_1).await.unwrap();
//         assert_eq!(
//             service
//                 .list(None, None, None, None, None)
//                 .await
//                 .unwrap()
//                 .len(),
//             2
//         );

//         // Verify remaining workspaces
//         let workspaces = service.list(None, None, None, None, None).await.unwrap();
//         assert_eq!(workspaces.len(), 2);

//         // Check first workspace in list
//         let first_workspace = workspaces.first().unwrap();
//         assert_eq!(first_workspace.id, id_2);
//         assert_eq!(first_workspace.workspace_name, "workspace2");
//         assert_eq!(first_workspace.bot_token, "xoxb-test-token-2");
//         assert_eq!(first_workspace.channels, Some(vec!["C2".to_string()]));
//     }

//     #[tokio::test]
//     async fn test_list_with_ids_filter() {
//         let db = DbConn::new_in_memory().await.unwrap();
//         let job = Arc::new(crate::service::job::create(db.clone()).await);
//         let service = create(db.clone(), job);

//         // Create multiple workspaces
//         let id_1 = service
//             .create(
//                 "workspace1".into(),
//                 "xoxb-test-token-1".into(),
//                 Some(vec!["C1".into()]),
//             )
//             .await
//             .unwrap();

//         let id_2 = service
//             .create(
//                 "workspace2".into(),
//                 "xoxb-test-token-2".into(),
//                 Some(vec!["C2".into()]),
//             )
//             .await
//             .unwrap();

//         // Test filtering by specific IDs
//         let filtered = service
//             .list(Some(vec![id_1.clone()]), None, None, None, None)
//             .await
//             .unwrap();
//         assert_eq!(filtered.len(), 1);
//         assert_eq!(filtered[0].id, id_1);

//         // Test filtering by multiple IDs
//         let filtered = service
//             .list(Some(vec![id_1, id_2]), None, None, None, None)
//             .await
//             .unwrap();
//         assert_eq!(filtered.len(), 2);
//     }

//     #[tokio::test]
//     async fn test_list_workspaces() {
//         let db = DbConn::new_in_memory().await.unwrap();
//         let job = Arc::new(crate::service::job::create(db.clone()).await);
//         let service = create(db.clone(), job);

//         // Create a few workspaces
//         service
//             .create(
//                 "workspace1".into(),
//                 "xoxb-test-token-1".into(),
//                 Some(vec!["C1".into()]),
//             )
//             .await
//             .unwrap();

//         service
//             .create(
//                 "workspace2".into(),
//                 "xoxb-test-token-2".into(),
//                 Some(vec!["C2".into()]),
//             )
//             .await
//             .unwrap();

//         // Test list_workspaces method
//         let workspaces = service.list_workspaces().await.unwrap();
//         assert_eq!(workspaces.len(), 2);
//         assert_eq!(workspaces[0].workspace_name, "workspace1");
//     }
// }
