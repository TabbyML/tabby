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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Choice {
    pub index: u32,
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum SelectKind {
    Line,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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
        user_agent: Option<String>,
    },
    ChatCompletion {
        completion_id: String,
        input: Vec<Message>,
        output: Message,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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

    #[serde(skip_serializing_if = "Option::is_none")]
    pub filepath: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Declaration {
    pub filepath: String,
    pub body: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LogEntry {
    pub user: Option<String>,
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

pub trait EventLogger: Send + Sync {
    fn log(&self, user: Option<String>, event: Event) {
        self.write(LogEntry {
            user,
            ts: timestamp(),
            event,
        })
    }
    fn write(&self, x: LogEntry);
}

pub struct ComposedLogger<T1: EventLogger, T2: EventLogger> {
    logger1: T1,
    logger2: T2,
}

impl<T1: EventLogger, T2: EventLogger> ComposedLogger<T1, T2> {
    pub fn new(logger1: T1, logger2: T2) -> Self {
        Self { logger1, logger2 }
    }
}

impl<T1: EventLogger, T2: EventLogger> EventLogger for ComposedLogger<T1, T2> {
    fn write(&self, x: LogEntry) {
        self.logger1.write(x.clone());
        self.logger2.write(x);
    }
}
