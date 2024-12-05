mod serper;
mod tantivy;

use std::sync::Arc;

use tabby_common::api::structured_doc::DocSearch;
use tabby_inference::Embedding;

use super::tantivy::IndexReaderProvider;

pub fn create(embedding: Arc<dyn Embedding>, provider: Arc<IndexReaderProvider>) -> impl DocSearch {
    tantivy::DocSearchService::new(embedding, provider)
}

pub fn create_serper(api_key: &str) -> impl DocSearch {
    serper::SerperService::new(api_key)
}
