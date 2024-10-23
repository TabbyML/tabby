use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, TimeZone, Utc};
use hyper::header;
use logkit::debug;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tabby_index::public::{DocIndexer, WebDocument};
use tabby_inference::Embedding;
use tabby_schema::CoreError;

use super::helper::Job;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SlackMessage {
    pub id: String,
    //unique id for the message
    pub ts: String,
    pub channel_id: String,
    pub user: String,
    pub text: String,
    pub timestamp: DateTime<Utc>,
    pub thread_ts: Option<String>,
    pub reply_users_count: Option<i32>,
    pub reply_count: Option<i32>,
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

//TODO: Implement these functions
async fn fetch_all_channels(
    bot_token: &str,
    workspace_id: &str,
) -> Result<Vec<SlackChannel>, CoreError> {
    debug!("unimplemented: fetch_all_channels");
    Ok(vec![])
}

async fn fetch_channel_messages(
    bot_token: &str,
    workspace_id: &str,
    channel_id: &str,
) -> Result<Vec<SlackMessage>, CoreError> {
    debug!("unimplemented: fetch_channel_messages");
    Ok(vec![])
}

async fn fetch_message_replies(
    bot_token: &str,
    channel_id: &str,
    thread_ts: &str,
) -> Result<Vec<SlackReply>, CoreError> {
    debug!("unimplemented: fetch_message_replies");
    Ok(vec![])
}

// Now it is utils for slack client

#[derive(Debug, Deserialize)]
struct Topic {
    value: String,
}

#[derive(Debug, Deserialize)]
struct Purpose {
    value: String,
}

#[derive(Debug, Deserialize)]
struct SlackChannelResponse {
    id: String,
    name: String,
    is_channel: bool,
    created: i64,
    is_archived: bool,
    is_general: bool,
    is_member: bool,
    is_private: bool,
    topic: Topic,
    purpose: Purpose,
    num_members: i32,
}

#[derive(Debug, Deserialize)]
struct ResponseMetadata {
    next_cursor: String,
}

#[derive(Debug, Deserialize)]
struct SlackResponse {
    ok: bool,
    channels: Option<Vec<SlackChannelResponse>>,
    error: Option<String>,
    response_metadata: Option<ResponseMetadata>,
}

/// Messages structs
#[derive(Debug, Deserialize)]
struct SlackMessageResponse {
    ok: bool,
    messages: Option<Vec<SlackMessageItemResponse>>,
    error: Option<String>,
    has_more: bool,
}

#[derive(Debug, Deserialize)]
struct SlackMessageItemResponse {
    #[serde(default)]
    subtype: Option<String>,
    user: String,
    text: String,
    r#type: String,
    ts: String,
    #[serde(default)]
    client_msg_id: Option<String>,
    #[serde(default)]
    team: Option<String>,
    //only exists in thread messages
    thread_ts: Option<String>,
    reply_users_count: Option<i32>,
    reply_count: Option<i32>,
}

// TODO implement slack basic client
struct SlackClient {
    bot_token: String,
    client: Client,
}
impl SlackClient {
    fn new(bot_token: &str) -> Result<Self, CoreError> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/x-www-form-urlencoded"),
        );

        let client: Client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;
        Ok(Self {
            bot_token: bot_token.to_string(),
            client,
        })
    }

    pub async fn get_channels(&self, _workspace_id: &str) -> Result<Vec<SlackChannel>> {
        let response = self
            .client
            .get("https://slack.com/api/conversations.list")
            .header(header::AUTHORIZATION, format!("Bearer {}", self.bot_token))
            .query(&[
                ("types", "public_channel"),
                ("exclude_archived", "true"),
                ("limit", "1000"),
            ])
            .send()
            .await?;

        let slack_response: SlackResponse = response.json().await?;

        match (
            slack_response.ok,
            slack_response.channels,
            slack_response.error,
        ) {
            (true, Some(channels), _) => {
                debug!("Successfully fetched {} channels", channels.len());
                Ok(channels
                    .into_iter()
                    .map(|s| SlackChannel {
                        id: s.id,
                        name: s.name,
                    })
                    .collect())
            }
            (false, _, Some(error)) => Err(anyhow::anyhow!("Slack API error: {}", error)),
            _ => Err(anyhow::anyhow!("Unexpected response from Slack API")),
        }
    }

    pub async fn get_messages(&self, channel_id: &str) -> Result<Vec<SlackMessage>, CoreError> {
        let response: reqwest::Response = self
            .client
            .get("https://slack.com/api/conversations.history")
            .header(header::AUTHORIZATION, format!("Bearer {}", self.bot_token))
            .query(&[
                ("channel", channel_id),
                ("limit", "100"),
                ("inclusive", "true"),
            ])
            .send()
            .await
            .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;

        let slack_response: SlackMessageResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;

        match (
            slack_response.ok,
            slack_response.messages,
            slack_response.error,
        ) {
            (true, Some(messages), _) => {
                debug!("Successfully fetched {} messages", messages.len());
                Ok(messages
                    .into_iter()
                    .filter(|msg| msg.subtype.is_none())
                    .map(|msg| {
                        let timestamp = msg
                            .ts
                            .split('.')
                            .next()
                            .and_then(|ts| ts.parse::<i64>().ok())
                            .map(|ts| Utc.timestamp_opt(ts, 0).unwrap())
                            .unwrap_or_default();

                        SlackMessage {
                            id: msg.ts.clone(),
                            ts: msg.ts.clone(),
                            channel_id: channel_id.to_string(),
                            user: msg.user,
                            text: msg.text,
                            thread_ts: msg.thread_ts,
                            reply_count: msg.reply_count,
                            reply_users_count: msg.reply_users_count,
                            replies: vec![],
                            timestamp,
                        }
                    })
                    .collect())
            }
            (false, _, Some(error)) => Err(CoreError::Other(anyhow::anyhow!(
                "Slack API error: {}",
                error
            ))),
            _ => Err(CoreError::Other(anyhow::anyhow!(
                "Unexpected response from Slack API"
            ))),
        }
    }
    pub async fn join_channels(&self, channel_ids: Vec<&str>) -> Result<bool, CoreError> {
        for channel_id in channel_ids {
            let response = self
                .client
                .post("https://slack.com/api/conversations.join")
                .header(header::AUTHORIZATION, format!("Bearer {}", self.bot_token))
                .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                .form(&[("channel", channel_id)])
                .send()
                .await
                .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;

            let slack_response: SlackResponse = response
                .json()
                .await
                .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;

            match (slack_response.ok, slack_response.error) {
                (false, Some(error)) => {
                    return Err(CoreError::Other(anyhow::anyhow!(
                        "Failed to join channel {}: {}",
                        channel_id,
                        error
                    )))
                }
                (false, None) => {
                    return Err(CoreError::Other(anyhow::anyhow!(
                        "Unexpected response from Slack API when joining channel {}",
                        channel_id
                    )))
                }
                _ => continue,
            }
        }

        Ok(true)
    }
}
