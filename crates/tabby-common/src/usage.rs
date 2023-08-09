use std::fs;

use lazy_static::lazy_static;
use reqwest::Client;
use serde::Serialize;
use uuid::Uuid;

use crate::path::usage_id_file;

static USAGE_API_ENDPOINT: &str = "https://app.tabbyml.com/api/usage";

struct UsageTracker {
    id: String,
    client: Option<Client>,
}

impl UsageTracker {
    fn new() -> Self {
        if fs::metadata(usage_id_file()).is_err() {
            // usage id file doesn't exists.
            let id = Uuid::new_v4().to_string();
            std::fs::write(usage_id_file(), id).expect("Failed to create usage id");
        }

        let id = fs::read_to_string(usage_id_file()).expect("Failed to read usage id");
        let client = if std::env::var("TABBY_DISABLE_USAGE_COLLECTION").is_ok() {
            None
        } else {
            Some(Client::new())
        };

        Self { id, client }
    }

    async fn capture<T>(&self, event: &str, properties: T)
    where
        T: Serialize,
    {
        if let Some(client) = &self.client {
            let payload = Payload {
                distinct_id: self.id.as_ref(),
                event,
                properties,
            };
            client
                .post(USAGE_API_ENDPOINT)
                .json(&payload)
                .send()
                .await
                .ok();
        }
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
    static ref TRACKER: UsageTracker = UsageTracker::new();
}

pub async fn capture<T>(event: &str, properties: T)
where
    T: Serialize,
{
    TRACKER.capture(event, properties).await
}
