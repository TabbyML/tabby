pub mod client;

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Utc;
use client::{SlackClient, SlackMessage, SlackReply};
use serde::{Deserialize, Serialize};
use tabby_index::public::{DocIndexer, WebDocument};
use tabby_inference::Embedding;
use tabby_schema::{
    job::JobService,
    slack_workspaces::{SlackChannel, SlackWorkspaceService},
    CoreError,
};
use tracing::{debug, info};

use super::{helper::Job, BackgroundJobEvent};

#[derive(Debug, Serialize, Deserialize)]
pub struct SlackIntegrationJob {
    pub source_id: String,
    pub bot_token: String,
    pub workspace_name: String,
    pub channels: Option<Vec<String>>,
    #[serde(skip)]
    client: SlackClient,
}

impl SlackIntegrationJob {
    pub async fn new(
        source_id: String,
        workspace_name: String,
        bot_token: String,
        channels: Option<Vec<String>>,
    ) -> Self {
        // Initialize the Slack client first
        let client = SlackClient::new(bot_token.clone())
            .await
            .map_err(|e| {
                debug!(
                    "Failed to initialize Slack client for workspace '{}': {:?}",
                    workspace_name, e
                );
                CoreError::Unauthorized("Slack client initialization failed")
            })
            .unwrap();

        Self {
            source_id,
            bot_token,
            workspace_name,
            channels,
            client,
        }
    }

    async fn ensure_client(&mut self) -> Result<(), CoreError> {
        debug!("Ensuring client with token: {}", self.bot_token);
        self.client = SlackClient::new(self.bot_token.clone()).await?;
        Ok(())
    }
}

impl Job for SlackIntegrationJob {
    const NAME: &'static str = "slack_integration";
}

impl SlackIntegrationJob {
    pub async fn run(mut self, embedding: Arc<dyn Embedding>) -> Result<(), CoreError> {
        info!(
            "Starting Slack integration for workspace: {}",
            self.workspace_name
        );
        self.ensure_client().await?;

        let mut num_indexed_messages = 0;
        let indexer = DocIndexer::new(embedding);

        // If specific channels are specified, join them first
        if let Some(channel_ids) = &self.channels {
            self.client
                .join_channels(channel_ids.iter().map(|s| s.as_str()).collect())
                .await
                .map_err(|e| {
                    debug!("Failed to join channels: {:?}", e);
                    e
                })?;
        }

        // Fetch and filter channels
        let channels = fetch_all_channels(&self.client).await.map_err(|e| {
            debug!("Failed to fetch channels: {:?}", e);
            e
        })?;

        let channels = if let Some(channel_ids) = &self.channels {
            channels
                .into_iter()
                .filter(|c| channel_ids.contains(&c.id))
                .collect::<Vec<_>>()
        } else {
            channels
        };

        debug!("Processing {} channels", channels.len());

        // Process each channel
        for channel in channels {
            debug!("Processing channel: {}", channel.name);
            let messages = fetch_channel_messages(&self.client, &channel.id)
                .await
                .map_err(|e| {
                    debug!(
                        "Failed to fetch messages for channel {}: {:?}",
                        channel.name, e
                    );
                    e
                })?
                .into_iter()
                .filter(|message| message.thread_ts.is_some())
                .collect::<Vec<_>>();

            for mut message in messages {
                // Fetch replies if thread exists
                let thread_ts = message.thread_ts.as_ref().unwrap();
                message.replies = fetch_message_replies(&self.client, &channel.id, thread_ts)
                    .await
                    .map_err(|e| {
                        debug!(
                            "Failed to fetch replies for message in channel {}: {:?}",
                            channel.name, e
                        );
                        e
                    })?;

                if should_index_message(&message) {
                    let web_doc = self.create_web_document(&channel, &message);
                    num_indexed_messages += 1;
                    debug!("Indexing message: {}", &web_doc.title);
                    indexer.add(Utc::now(), web_doc).await;
                }
            }
        }

        info!(
            "Indexed {} messages from Slack workspace '{}'",
            num_indexed_messages, self.workspace_name
        );
        indexer.commit();
        Ok(())
    }

    /// cron job to sync slack messages
    pub async fn cron(
        slack_workspace: Arc<dyn SlackWorkspaceService>,
        job: Arc<dyn JobService>,
    ) -> tabby_schema::Result<()> {
        let workspaces = slack_workspace
            .list_workspaces()
            .await
            .context("Must be able to retrieve slack workspace for sync")?;

        for workspace in workspaces {
            let _ = job
                .trigger(
                    BackgroundJobEvent::SlackIntegration(
                        SlackIntegrationJob::new(
                            workspace.id.to_string(),
                            workspace.workspace_name.clone(),
                            workspace.bot_token.clone(),
                            workspace.channels.clone(),
                        )
                        .await,
                    )
                    .to_command(),
                )
                .await;
        }
        Ok(())
    }

    /// Create a WebDocument for a Slack message with replies
    fn create_web_document(&self, channel: &SlackChannel, message: &SlackMessage) -> WebDocument {
        let mut content = message.text.clone();
        for reply in &message.replies {
            content.push_str("\n\nReply: ");
            content.push_str(&reply.text);
        }

        WebDocument {
            source_id: self.source_id.clone(),
            id: format!("{}:{}", channel.id, message.id),
            title: format!(
                "Slack message in #{} with message id {}",
                channel.name, message.id
            ),
            link: format!(
                "https://slack.com/archives/{}/p{}",
                channel.id,
                message.id.replace(".", "")
            ),
            body: content,
        }
    }
}

/// the index message should be long enough and have replies
fn should_index_message(message: &SlackMessage) -> bool {
    message.text.len() > 80 && message.reply_count.is_some()
}

async fn fetch_all_channels(client: &SlackClient) -> Result<Vec<SlackChannel>, CoreError> {
    client.get_channels().await.map_err(CoreError::Other)
}

async fn fetch_channel_messages(
    client: &SlackClient,
    channel_id: &str,
) -> Result<Vec<SlackMessage>, CoreError> {
    client.get_messages(channel_id).await
}

async fn fetch_message_replies(
    client: &SlackClient,
    channel_id: &str,
    thread_ts: &str,
) -> Result<Vec<SlackReply>, CoreError> {
    client.get_message_replies(channel_id, thread_ts).await
}

#[cfg(test)]
mod tests {

    use super::*;

    #[tokio::test]
    async fn test_should_index_message() {
        // not index because no replies
        let message = SlackMessage {
            id: "1".to_string(),
            ts: "1234567890.123456".to_string(),
            channel_id: "C1234567890".to_string(),
            user: "U1234567890".to_string(),
            text: "A".repeat(81),
            timestamp: Utc::now(),
            thread_ts: None,
            reply_users_count: None,
            reply_count: None,
            replies: vec![],
        };

        assert!(!should_index_message(&message));

        // not index because message is too short
        let short_message = SlackMessage {
            text: "Short message".to_string(),
            ..message.clone()
        };

        assert!(!should_index_message(&short_message));

        // good message should be index
        let message_with_replies = SlackMessage {
            replies: vec![SlackReply {
                id: "1".to_string(),
                user: "U1234567890".to_string(),
                text: "this is approach to solve this question".to_string(),
                timestamp: Utc::now(),
                thread_ts: Some("asd".to_string()),
                reply_count: Some(2),
                subscribed: Some(true),
                last_read: Some("asd".to_string()),
                unread_count: Some(3),
                parent_user_id: Some("1".to_string()),
                r#type: "message".to_string(),
            }],
            reply_count: Some(2),
            ..message
        };

        assert!(should_index_message(&message_with_replies));
    }
}
