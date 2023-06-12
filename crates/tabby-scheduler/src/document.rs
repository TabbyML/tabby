use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Document {
    pub git_url: String,
    pub filepath: String,
    pub content: String,
    pub language: String,
}
