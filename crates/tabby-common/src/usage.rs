use std::{
    collections::HashMap,
    fs::{self},
};

use lazy_static::lazy_static;
use reqwest::Client;
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
        }

        let id = fs::read_to_string(usage_id_file()).expect("Failed to read usage id");
        let client = Client::new();

        Self { id, client }
    }

    async fn capture(&self, event: &str) {
        let params = HashMap::from([("distinctId", self.id.as_ref()), ("event", event)]);
        self.client
            .post(USAGE_API_ENDPOINT)
            .json(&params)
            .send()
            .await
            .ok();
    }
}

lazy_static! {
    static ref TRACKER: UsageTracker = UsageTracker::new();
}

pub async fn capture(event: &str) {
    TRACKER.capture(event).await
}

#[cfg(test)]
mod tests {
    use super::capture;

    #[tokio::test]
    async fn it_fire_event() {
        capture("UsageTest").await
    }
}
