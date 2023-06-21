pub mod config;
pub mod events;
pub mod path;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Document {
    pub git_url: String,
    pub filepath: String,
    pub content: String,
    pub language: String,
    pub max_line_length: usize,
    pub avg_line_length: f32,
    pub alphanum_fraction: f32,
}
