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
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
    pub technologies: Vec<String>,
    pub raw: String,
}

#[derive(Serialize)]
pub struct CrawledMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
}

impl From<readable_readability::Metadata> for CrawledMetadata {
    fn from(metadata: readable_readability::Metadata) -> Self {
        Self {
            title: metadata.article_title.or(metadata.page_title),
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
