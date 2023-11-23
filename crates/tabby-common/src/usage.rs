use std::fs;

use lazy_static::lazy_static;
use reqwest::Client;
use serde::Serialize;
use uuid::Uuid;

use crate::path::usage_id_file;

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

            eprintln!(
                "
  \x1b[34;1mTELEMETRY\x1b[0m

  As an open source project, we collect usage statistics to inform development priorities. For more
  information, read https://tabby.tabbyml.com/docs/configuration#usage-collection

  We will not see or any code in your development process.

  To opt-out, add the TABBY_DISABLE_USAGE_COLLECTION=1 to your tabby server's environment variables.

  \x1b[1mWelcome to Tabby!\x1b[0m

  If you have any questions or would like to engage with the Tabby team, please join us on Slack
  (https://tinyurl.com/35sv9kz2).

"
            );
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
