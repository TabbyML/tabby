use anyhow::Result;
use chrono::{DateTime, TimeZone, Utc};
use hyper::header;
use logkit::debug;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tabby_schema::CoreError;

// TODO: move types into slack mod
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SlackMessage {
    pub id: String,
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
    pub thread_ts: Option<String>,
    pub reply_count: Option<i32>,
    pub subscribed: Option<bool>,
    pub last_read: Option<String>,
    pub unread_count: Option<i32>,
    pub parent_user_id: Option<String>,
    pub r#type: String,
}
#[derive(Debug, Clone)]
pub struct SlackChannel {
    pub id: String,
    pub name: String,
}

// API Response Types
#[derive(Debug, Deserialize)]
pub(crate) struct SlackChannelApiResponse {
    pub id: String,
    pub name: String,
    pub is_channel: bool,
    pub created: i64,
    pub is_archived: bool,
    pub is_general: bool,
    pub is_member: bool,
    pub is_private: bool,
    pub topic: SlackChannelTopic,
    pub purpose: SlackChannelPurpose,
    pub num_members: i32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlackChannelTopic {
    pub value: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlackChannelPurpose {
    pub value: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlackListChannelsResponse {
    pub ok: bool,
    pub channels: Option<Vec<SlackChannelApiResponse>>,
    pub error: Option<String>,
    pub response_metadata: Option<SlackResponseMetadata>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlackResponseMetadata {
    pub next_cursor: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlackBasicResponse {
    pub ok: bool,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlackListMessagesResponse {
    pub ok: bool,
    pub messages: Option<Vec<SlackMessageApiResponse>>,
    pub error: Option<String>,
    pub has_more: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlackMessageApiResponse {
    #[serde(default)]
    pub subtype: Option<String>,
    pub user: String,
    pub text: String,
    pub r#type: String,
    pub ts: String,
    #[serde(default)]
    pub client_msg_id: Option<String>,
    #[serde(default)]
    pub team: Option<String>,
    pub thread_ts: Option<String>,
    pub reply_users_count: Option<i32>,
    pub reply_count: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlackAuthTestResponse {
    pub ok: bool,
    pub url: Option<String>,
    pub team: Option<String>,
    pub user: Option<String>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub bot_id: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlackMessageRepliesResponse {
    pub ok: bool,
    pub messages: Option<Vec<SlackThreadMessageResponse>>,
    pub has_more: bool,
    pub response_metadata: Option<SlackResponseMetadata>,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlackThreadMessageResponse {
    pub r#type: String,
    pub user: String,
    pub text: String,
    pub thread_ts: Option<String>,
    pub ts: String,
    pub reply_count: Option<i32>,
    pub subscribed: Option<bool>,
    pub last_read: Option<String>,
    pub unread_count: Option<i32>,
    pub parent_user_id: Option<String>,
}

/// Slack API client for making requests to Slack's Web API
#[derive(Debug, Clone)]
pub struct SlackClient {
    bot_token: String,
    client: Client,
}

impl Default for SlackClient {
    fn default() -> Self {
        Self {
            bot_token: "".to_string(),
            client: Client::new(),
        }
    }
}

impl SlackClient {
    pub async fn new(bot_token: &str) -> Result<Self, CoreError> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/x-www-form-urlencoded"),
        );

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;

        let slack_client = Self {
            bot_token: bot_token.to_string(),
            client,
        };

        // Validate token immediately
        let is_valid = slack_client.validate_token().await?;
        if !is_valid {
            return Err(CoreError::Unauthorized("Invalid Slack bot token"));
        }

        Ok(slack_client)
    }

    /// Fetches all channels from a Slack workspace
    pub async fn get_channels(&self) -> Result<Vec<SlackChannel>> {
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

        let api_response: SlackListChannelsResponse = response.json().await?;

        match (api_response.ok, api_response.channels, api_response.error) {
            (true, Some(channels), _) => {
                debug!("Successfully fetched {} channels", channels.len());
                Ok(channels
                    .into_iter()
                    .map(|channel| SlackChannel {
                        id: channel.id,
                        name: channel.name,
                    })
                    .collect())
            }
            (false, _, Some(error)) => Err(anyhow::anyhow!("Slack API error: {}", error)),
            _ => Err(anyhow::anyhow!("Unexpected response from Slack API")),
        }
    }

    /// Fetches messages from a specific channel
    pub async fn get_messages(&self, channel_id: &str) -> Result<Vec<SlackMessage>, CoreError> {
        let response = self
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

        let api_response: SlackListMessagesResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;

        match (api_response.ok, api_response.messages, api_response.error) {
            (true, Some(messages), _) => {
                debug!("Successfully fetched {} messages", messages.len());
                Ok(messages
                    .into_iter()
                    .filter(|msg| msg.subtype.is_none())
                    .map(|msg| convert_to_slack_message(msg, channel_id))
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

    /// Joins specified channels in the workspace
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

            let api_response: SlackBasicResponse = response
                .json()
                .await
                .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;

            match (api_response.ok, api_response.error) {
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

    /// Fetches all replies in a thread without parent message
    pub async fn get_message_replies(
        &self,
        channel_id: &str,
        thread_ts: &str,
    ) -> Result<Vec<SlackReply>, CoreError> {
        let response = self
            .client
            .get("https://slack.com/api/conversations.replies")
            .header(header::AUTHORIZATION, format!("Bearer {}", self.bot_token))
            .query(&[
                ("channel", channel_id),
                ("ts", thread_ts),
                ("limit", "1000"),
                ("inclusive", "true"),
            ])
            .send()
            .await
            .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;

        let api_response: SlackMessageRepliesResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;

        match (api_response.ok, api_response.messages, api_response.error) {
            (true, Some(messages), _) => {
                debug!("Successfully fetched {} replies", messages.len());

                // Convert messages to SlackReply format, skipping the parent message
                let replies: Vec<SlackReply> = messages
                    .into_iter()
                    .skip(1) // Skip first parent message due to Slack API design
                    .map(convert_to_slack_reply)
                    .collect();

                Ok(replies)
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

    async fn validate_token(&self) -> Result<bool, CoreError> {
        let response = self
            .client
            .post("https://slack.com/api/auth.test")
            .header(header::AUTHORIZATION, format!("Bearer {}", self.bot_token))
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .send()
            .await
            .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;

        let auth_response: SlackAuthTestResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Other(anyhow::Error::new(e)))?;

        if !auth_response.ok {
            debug!("Token validation failed: {:?}", auth_response.error);
            return Ok(false);
        }

        debug!(
            "Token validated successfully. Connected as: {:?} to workspace: {:?}",
            auth_response.user, auth_response.team
        );

        Ok(true)
    }
}

fn convert_to_slack_message(msg: SlackMessageApiResponse, channel_id: &str) -> SlackMessage {
    let timestamp = msg
        .ts
        .split('.')
        .next()
        .and_then(|ts| ts.parse::<i64>().ok())
        .map(|ts| Utc.timestamp_opt(ts, 0).unwrap())
        .unwrap_or_default();

    SlackMessage {
        id: msg.ts.clone(),
        ts: msg.ts,
        channel_id: channel_id.to_string(),
        user: msg.user,
        text: msg.text,
        thread_ts: msg.thread_ts,
        reply_count: msg.reply_count,
        reply_users_count: msg.reply_users_count,
        replies: vec![],
        timestamp,
    }
}

fn convert_to_slack_reply(msg: SlackThreadMessageResponse) -> SlackReply {
    let timestamp = msg
        .ts
        .split('.')
        .next()
        .and_then(|ts| ts.parse::<i64>().ok())
        .map(|ts| Utc.timestamp_opt(ts, 0).unwrap())
        .unwrap_or_default();

    SlackReply {
        id: msg.ts,
        user: msg.user,
        text: msg.text,
        timestamp,
        thread_ts: msg.thread_ts,
        reply_count: msg.reply_count,
        subscribed: msg.subscribed,
        last_read: msg.last_read,
        unread_count: msg.unread_count,
        parent_user_id: msg.parent_user_id,
        r#type: msg.r#type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_channel_ids() {
        let Ok(client) = SlackClient::new("your-bot-token").await else {
            // test
            return;
        };

        let channels = client.get_channels().await.unwrap();
        debug!("{:?}", channels);
    }

    #[tokio::test]
    async fn test_get_messages() {
        let Ok(client) = SlackClient::new("your-bot-token").await else {
            // test
            return;
        };
        let messages = client.get_messages("channel-id").await.unwrap();
        debug!("{:?}", messages);
    }
    #[tokio::test]
    async fn test_get_replies() {
        let Ok(client) = SlackClient::new("your-bot-token").await else {
            // test
            return;
        };
        //1729751729.608689
        let result = client
            .get_message_replies("channel-id", "ts")
            .await
            .unwrap();
        debug!("{:?}", result);
    }

    #[tokio::test]
    async fn test_join_channels() {
        let Ok(client) = SlackClient::new("your-bot-token").await else {
            // test
            return;
        };
        let result = client
            .join_channels(vec!["channel-id-1", "channel-id-2"])
            .await
            .unwrap();
        debug!("{:?}", result);
    }
}
