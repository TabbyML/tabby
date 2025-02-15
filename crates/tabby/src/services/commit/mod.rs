mod tantivy;

use std::sync::Arc;

use tabby_common::api::commit::CommitHistorySearch;
use tabby_inference::Embedding;

use super::tantivy::IndexReaderProvider;

pub fn create(
    embedding: Arc<dyn Embedding>,
    provider: Arc<IndexReaderProvider>,
) -> impl CommitHistorySearch {
    tantivy::new(embedding, provider)
}
