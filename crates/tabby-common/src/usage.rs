use std::fs;

use lazy_static::lazy_static;
use reqwest::Client;
use serde::Serialize;
use uuid::Uuid;

use crate::{
    path::usage_id_file,
    terminal::{HeaderFormat, InfoMessage},
};

static USAGE_API_ENDPOINT: &str = "https://app.tabbyml.com/api/usage";

struct UsageTracker {
    id: String,
    client: Client,
}

impl UsageTracker {
    fn new() -> Self {
        if fs::metadata(usage_id_file()).is_err() {
            // usage id file doesn't exists.
            let id = Uuid::new_v4().to_string();
            std::fs::write(usage_id_file(), id).expect("Failed to create usage id");

            InfoMessage::print_messages(&[
                InfoMessage::new("TELEMETRY", HeaderFormat::BoldBlue, &[
                    "As an open source project, we collect usage statistics to inform development priorities. For more",
                    "information, read https://tabby.tabbyml.com/docs/configuration#usage-collection",
                    "",
                    "We will not see or collect any code in your development process."
                ]),
                InfoMessage::new("Welcome to Tabby!", HeaderFormat::BoldWhite, &[
                    "If you have any questions or would like to engage with the Tabby team, please join us on Slack",
                    "(https://links.tabbyml.com/join-slack-terminal)."
                ])
            ]);
        }

        let id = fs::read_to_string(usage_id_file()).expect("Failed to read usage id");
        Self {
            id,
            client: Client::new(),
        }
    }

    async fn capture<T>(&self, event: &str, properties: T)
    where
        T: Serialize,
    {
        let payload = Payload {
            distinct_id: &self.id,
            event,
            properties,
        };
        self.client
            .post(USAGE_API_ENDPOINT)
            .json(&payload)
            .send()
            .await
            .ok();
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Payload<'a, T> {
    distinct_id: &'a str,
    event: &'a str,
    properties: T,
}

lazy_static! {
    static ref TRACKER: Option<UsageTracker> = {
        if std::env::var("TABBY_DISABLE_USAGE_COLLECTION").is_ok() {
            None
        } else {
            Some(UsageTracker::new())
        }
    };
}

pub async fn capture<T>(event: &str, properties: T)
where
    T: Serialize,
{
    if let Some(tracker) = TRACKER.as_ref() {
        tracker.capture(event, properties).await;
    }
}
