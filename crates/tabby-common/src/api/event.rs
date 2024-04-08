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

#[derive(Serialize, Deserialize, Debug)]
pub struct Choice {
    pub index: u32,
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum SelectKind {
    Line,
}

#[derive(Serialize, Deserialize, Debug)]
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

#[derive(Serialize, Deserialize, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Segments {
    pub prefix: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clipboard: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_url: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub declarations: Option<Vec<Declaration>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Declaration {
    pub filepath: String,
    pub body: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LogEntry {
    pub ts: u128,
    pub event: Event,
}

impl From<Event> for LogEntry {
    fn from(event: Event) -> Self {
        Self {
            ts: timestamp(),
            event,
        }
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

pub trait EventLogger: Send + Sync {
    fn log(&self, x: Event) {
        self.write(x.into())
    }
    fn write(&self, x: LogEntry);
}
