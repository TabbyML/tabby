use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct LogEventRequest {
    /// Event type, should be `view` or `select`.
    #[schema(example = "view")]
    #[serde(rename = "type")]
    pub event_type: String,

    pub completion_id: String,

    pub choice_index: u32,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: u32,
    pub text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectKind {
    Line,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    View {
        completion_id: String,
        choice_index: u32,
    },
    Select {
        completion_id: String,
        choice_index: u32,

        #[serde(skip_serializing_if = "Option::is_none")]
        kind: Option<SelectKind>,
    },
    Completion {
        completion_id: String,
        language: String,
        prompt: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        segments: Option<Segments>,
        choices: Vec<Choice>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user: Option<String>,
    },
}

#[derive(Serialize)]
pub struct Segments {
    pub prefix: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clipboard: Option<String>,
}

pub trait EventLogger: Send + Sync {
    fn log(&self, e: Event);
}

#[derive(Serialize)]
struct Log {
    ts: u128,
    event: Event,
}

pub trait RawEventLogger: Send + Sync {
    fn log(&self, content: String);
}

impl<T: RawEventLogger> EventLogger for T {
    fn log(&self, e: Event) {
        let content = serdeconv::to_json_string(&Log {
            ts: timestamp(),
            event: e,
        })
        .unwrap();

        self.log(content);
    }
}

fn timestamp() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let start = SystemTime::now();
    start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis()
}
