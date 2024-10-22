use std::sync::Arc;

use chrono::{DateTime, Utc};
use logkit::debug;
use serde::{Deserialize, Serialize};
use tabby_index::public::{DocIndexer, WebDocument};
use tabby_inference::Embedding;
use tabby_schema::CoreError;

use super::helper::Job;

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

//TODO implement slack basic client
struct SlackClient {}
