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
}

pub trait EventLogger: Send + Sync {
    fn log(&self, e: &Event);
}
