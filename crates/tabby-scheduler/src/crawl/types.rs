use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
pub struct KatanaRequestResponse {
    pub timestamp: String,
    pub request: KatanaRequest,
    pub response: KatanaResponse,
}

#[derive(Deserialize, Debug)]
pub struct KatanaRequest {
    pub method: String,
    pub endpoint: String,
    pub raw: String,
}

#[derive(Deserialize, Debug)]
pub struct KatanaResponse {
    pub status_code: Option<u16>,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
    pub technologies: Option<Vec<String>>,
    pub raw: Option<String>,
}

#[derive(Serialize)]
pub struct CrawledMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
}

impl From<readable_readability::Metadata> for CrawledMetadata {
    fn from(metadata: readable_readability::Metadata) -> Self {
        // Trim all ascii special chars from title
        let trim_title_chars = [
            '#', '$', '%', '&', '*', '+', ',', '/', ':', ';', '=', '?', '@', '[', ']', '^', '`',
            '{', '|', '}', '~', '\n', ' ',
        ];
        let title = metadata
            .article_title
            .or(metadata.page_title)
            .map(|x| x.trim_matches(trim_title_chars).to_owned());
        Self {
            title,
            description: metadata.description,
        }
    }
}

#[derive(Serialize)]
pub struct CrawledDocument {
    pub url: String,
    pub markdown: String,

    pub metadata: CrawledMetadata,
}

impl CrawledDocument {
    pub fn new(url: String, markdown: String, metadata: CrawledMetadata) -> Self {
        Self {
            url,
            markdown,
            metadata,
        }
    }
}
