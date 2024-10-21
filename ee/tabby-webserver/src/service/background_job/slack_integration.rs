use super::helper::Job;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tabby_index::public::{DocIndexer, WebDocument};
use tabby_inference::Embedding;
use tabby_schema::CoreError;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SlackMessage {
    pub id: String,
    pub channel_id: String,
    pub user: String,
    pub text: String,
    pub timestamp: DateTime<Utc>,
    pub thread_ts: Option<String>,
    pub replies: Vec<SlackReply>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SlackReply {
    pub id: String,
    pub user: String,
    pub text: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SlackIntegrationJob {
    pub source_id: String,
    pub workspace_id: String,
    pub bot_token: String,
    //if none, index all channels, else index only the specified channels
    pub channels: Option<Vec<String>>,
}

impl Job for SlackIntegrationJob {
    const NAME: &'static str = "slack_integration";
}

impl SlackIntegrationJob {
    pub fn new(
        source_id: String,
        workspace_id: String,
        bot_token: String,
        channels: Option<Vec<String>>,
    ) -> Self {
        Self {
            source_id,
            workspace_id,
            bot_token,
            channels,
        }
    }

    pub async fn run(self, embedding: Arc<dyn Embedding>) -> tabby_schema::Result<()> {
        logkit::info!(
            "Starting Slack integration for workspace {}",
            self.workspace_id
        );
        let embedding = embedding.clone();
        let mut num_indexed_messages = 0;
        let indexer = DocIndexer::new(embedding);

        let channels = fetch_all_channels(&self.bot_token, &self.workspace_id).await?;

        for channel in channels {
            let messages =
                fetch_channel_messages(&self.bot_token, &self.workspace_id, &channel.id).await?;

            for message in messages {
                if should_index_message(&message) {
                    let web_doc = self.create_web_document(&channel, &message);

                    num_indexed_messages += 1;
                    logkit::debug!("Indexing message: {}", &web_doc.title);
                    indexer.add(message.timestamp, web_doc).await;
                }
            }
        }

        logkit::info!(
            "Indexed {} messages from Slack workspace '{}'",
            num_indexed_messages,
            self.workspace_id
        );
        indexer.commit();
        Ok(())
    }

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
                "Slack message in #{} at {}",
                channel.name, message.timestamp
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

fn should_index_message(message: &SlackMessage) -> bool {
    message.text.len() > 80 || !message.replies.is_empty()
}

#[derive(Debug, Clone)]
struct SlackChannel {
    id: String,
    name: String,
}

async fn fetch_all_channels(
    bot_token: &str,
    workspace_id: &str,
) -> Result<Vec<SlackChannel>, CoreError> {
    let client = slack_api::default_client().map_err(|e| CoreError::Other(e.into()))?;

    let mut channels = Vec::new();
    let mut cursor = None;

    loop {
        let request = ListRequest {
            token: bot_token.to_string(),
            cursor: cursor.clone(),
            exclude_archived: Some(true),
            types: Some("public_channel,private_channel".to_string()),
            ..Default::default()
        };

        let response = slack_api::channels::list(&client, &request)
            .await
            .map_err(|e| CoreError::Other(e.into()))?;

        if let Some(channel_list) = response.channels {
            for channel in channel_list {
                channels.push(SlackChannel {
                    id: channel.id.unwrap_or_default(),
                    name: channel.name.unwrap_or_default(),
                });
            }
        }

        if let Some(next_cursor) = response.response_metadata.and_then(|m| m.next_cursor) {
            if next_cursor.is_empty() {
                break;
            }
            cursor = Some(next_cursor);
        } else {
            break;
        }
    }

    Ok(channels)
}

async fn fetch_channel_messages(
    bot_token: &str,
    workspace_id: &str,
    channel_id: &str,
) -> Result<Vec<SlackMessage>, CoreError> {
    let client = slack_api::default_client().map_err(|e| CoreError::Other(e.into()))?;

    let mut messages = Vec::new();
    let mut cursor = None;

    loop {
        let request = HistoryRequest {
            token: bot_token.to_string(),
            channel: channel_id.to_string(),
            cursor: cursor.clone(),
            limit: Some(100),
            ..Default::default()
        };

        let response = slack_api::conversations::history(&client, &request)
            .await
            .map_err(|e| CoreError::Other(e.into()))?;

        if let Some(message_list) = response.messages {
            for msg in message_list {
                let replies = if msg.thread_ts.is_some() {
                    fetch_message_replies(bot_token, channel_id, &msg.ts.unwrap_or_default())
                        .await?
                } else {
                    Vec::new()
                };

                messages.push(SlackMessage {
                    id: msg.ts.unwrap_or_default(),
                    channel_id: channel_id.to_string(),
                    user: msg.user.unwrap_or_default(),
                    text: msg.text.unwrap_or_default(),
                    timestamp: chrono::DateTime::parse_from_str(
                        &msg.ts.unwrap_or_default(),
                        "%s%.f",
                    )
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| CoreError::Other(e.into()))?,
                    thread_ts: msg.thread_ts,
                    replies,
                });
            }
        }

        if let Some(next_cursor) = response.response_metadata.and_then(|m| m.next_cursor) {
            if next_cursor.is_empty() {
                break;
            }
            cursor = Some(next_cursor);
        } else {
            break;
        }
    }

    Ok(messages)
}

async fn fetch_message_replies(
    bot_token: &str,
    channel_id: &str,
    thread_ts: &str,
) -> Result<Vec<SlackReply>, CoreError> {
    let client = slack_api::default_client().map_err(|e| CoreError::Other(e.into()))?;

    let request = slack_api::conversations::RepliesRequest {
        token: bot_token.to_string(),
        channel: channel_id.to_string(),
        ts: thread_ts.to_string(),
        ..Default::default()
    };

    let response = slack_api::conversations::replies(&client, &request)
        .await
        .map_err(|e| CoreError::Other(e.into()))?;

    let mut replies = Vec::new();

    if let Some(message_list) = response.messages {
        // Skip the first message as it's the parent message
        for msg in message_list.into_iter().skip(1) {
            replies.push(SlackReply {
                id: msg.ts.unwrap_or_default(),
                user: msg.user.unwrap_or_default(),
                text: msg.text.unwrap_or_default(),
                timestamp: chrono::DateTime::parse_from_str(&msg.ts.unwrap_or_default(), "%s%.f")
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| CoreError::Other(e.into()))?,
            });
        }
    }

    Ok(replies)
}
