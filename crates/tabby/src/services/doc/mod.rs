mod serper;
mod tantivy;

use std::sync::Arc;

use tabby_common::api::doc::DocSearch;
use tabby_inference::Embedding;

pub fn create(embedding: Arc<dyn Embedding>) -> impl DocSearch {
    tantivy::DocSearchService::new(embedding)
}

pub fn create_serper(api_key: &str) -> impl DocSearch {
    serper::SerperService::new(api_key)
}
