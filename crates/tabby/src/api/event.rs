



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
