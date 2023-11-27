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
pub struct Choice<'a> {
    pub index: u32,
    pub text: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectKind {
    Line,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Event<'a> {
    View {
        completion_id: &'a str,
        choice_index: u32,
    },
    Select {
        completion_id: &'a str,
        choice_index: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        kind: Option<SelectKind>,
    },
    Completion {
        completion_id: &'a str,
        language: &'a str,
        prompt: &'a str,
        segments: &'a Option<Segments>,
        choices: Vec<Choice<'a>>,
        user: Option<&'a str>,
    },
}

#[derive(Serialize)]
pub struct Segments {
    pub prefix: String,
    pub suffix: Option<String>,
    pub clipboard: Option<String>,
}

pub trait EventLogger: Send + Sync {
    fn log(&self, e: &Event);
}

#[derive(Serialize)]
struct Log<'a> {
    ts: u128,
    event: &'a Event<'a>,
}

pub trait RawEventLogger: Send + Sync {
    fn log(&self, content: String);
}

impl<T: RawEventLogger> EventLogger for T {
    fn log(&self, e: &Event) {
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
