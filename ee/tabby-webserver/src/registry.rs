use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tabby_common::api::code::{BoxCodeSearch, CodeSearch, CodeSearchError, SearchResponse};

#[derive(Default)]
struct WorkerRegistry {
    code: Vec<Box<dyn CodeSearch>>,
}

impl WorkerRegistry {
    fn register(&mut self, code: BoxCodeSearch) {
        self.code.push(code)
    }
}

#[async_trait]
impl CodeSearch for WorkerRegistry {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        if self.code.is_empty() {
            Err(CodeSearchError::NotReady)
        } else {
            let code = &self.code[random_index(self.code.len())];
            code.search(q, limit, offset).await
        }
    }

    async fn search_with_query(
        &self,
        q: &dyn tantivy::query::Query,
        limit: usize,
        offset: usize,
    ) -> Result<SearchResponse, CodeSearchError> {
        if self.code.is_empty() {
            Err(CodeSearchError::NotReady)
        } else {
            let code = &self.code[random_index(self.code.len())];
            code.search_with_query(q, limit, offset).await
        }
    }
}

fn random_index(size: usize) -> usize {
    let unix_timestamp = (SystemTime::now().duration_since(UNIX_EPOCH))
        .unwrap()
        .as_nanos();
    let index = unix_timestamp % (size as u128);
    index as usize
}
