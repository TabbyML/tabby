use async_trait::async_trait;
use std::sync::Arc;
use tabby_common::api::{
    commit::{CommitHistorySearch, CommitHistorySearchParams, CommitHistorySearchResponse},
    Result, SearchError,
};
use tabby_inference::Embedding;

use crate::services::tantivy::IndexReaderProvider;

pub struct CommitHistorySearchImpl {
    embedding: Arc<dyn Embedding>,
    provider: Arc<IndexReaderProvider>,
}

pub fn new(
    embedding: Arc<dyn Embedding>,
    provider: Arc<IndexReaderProvider>,
) -> CommitHistorySearchImpl {
    CommitHistorySearchImpl {
        embedding,
        provider,
    }
}

#[async_trait]
impl CommitHistorySearch for CommitHistorySearchImpl {
    async fn search(
        &self,
        source_id: &str,
        content: &str,
        params: &CommitHistorySearchParams,
    ) -> Result<CommitHistorySearchResponse> {
        //FIXME(kweizh)
        Err(SearchError::NotReady)
    }
}
