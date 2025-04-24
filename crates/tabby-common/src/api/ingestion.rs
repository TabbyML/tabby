use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug, Validate)]
pub struct IngestionRequest {
    /// Source of the document (frontend available, backend sourceId: `ingestedSource:${source}`)
    #[validate(length(
        min = 1,
        max = 256,
        code = "source",
        message = "source can not be empty"
    ))]
    pub source: String,

    /// unique whthin the same source
    #[validate(length(min = 1, max = 256, code = "id", message = "id can not be empty"))]
    pub id: String,

    #[validate(length(
        min = 1,
        max = 65535,
        code = "title",
        message = "title can not be empty"
    ))]
    pub title: String,
    pub body: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,

    /// Time-to-live duration (optional). Duration string like "90d"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

#[derive(Serialize, Deserialize, ToSchema, Clone, Debug)]
pub struct IngestionResponse {
    pub id: String,
    pub source: String,
    pub message: String,
}
