use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct LogEventRequest {
    /// Event type, should be `view`, `select` or `dismiss`.
    #[schema(example = "view")]
    #[serde(rename = "type")]
    pub event_type: String,

    pub completion_id: String,

    pub choice_index: u32,

    pub view_id: Option<String>,

    pub elapsed: Option<u32>,
}

#[derive(Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub text: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectKind {
    Line,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    View {
        completion_id: String,
        choice_index: u32,

        #[serde(skip_serializing_if = "Option::is_none")]
        view_id: Option<String>,
    },
    Select {
        completion_id: String,
        choice_index: u32,

        #[serde(skip_serializing_if = "Option::is_none")]
        kind: Option<SelectKind>,

        #[serde(skip_serializing_if = "Option::is_none")]
        view_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        elapsed: Option<u32>,
    },
    Dismiss {
        completion_id: String,
        choice_index: u32,

        #[serde(skip_serializing_if = "Option::is_none")]
        view_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        elapsed: Option<u32>,
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
    ChatCompletion {
        completion_id: String,
        input: Vec<Message>,
        output: Message,
    },
}

#[derive(Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize)]
pub struct Segments {
    pub prefix: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clipboard: Option<String>,
}

pub trait EventLogger: Send + Sync {
    fn log(&self, e: Event) {
        let content = serdeconv::to_json_string(&Log {
            ts: timestamp(),
            event: e,
        })
        .unwrap();

        self.log_raw(content);
    }

    fn log_raw(&self, content: String);
}

#[derive(Serialize, Deserialize)]
pub struct Log {
    pub ts: u128,
    pub event: Event,
}

fn timestamp() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let start = SystemTime::now();
    start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis()
}
