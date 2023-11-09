mod juniper_axum;

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use axum::{
    routing::{get, post},
    Extension, Router,
};
use juniper::{graphql_object, EmptyMutation, EmptySubscription, RootNode};
use juniper_axum::{graphiql, graphql, playground};
use tabby_common::api::code::{CodeSearch, CodeSearchError, SearchResponse};

#[derive(Default)]
struct WorkerRegistry {
    code: Vec<Box<dyn CodeSearch>>,
}

impl WorkerRegistry {
    fn register(&mut self, code: Box<dyn CodeSearch>) {
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

#[derive(Clone, Copy, Debug)]
pub struct Query;

#[graphql_object]
impl Query {
    fn workers() -> i32 {
        100
    }

    fn add(a: i32, b: i32) -> i32 {
        a + b
    }
}

type Schema = RootNode<'static, Query, EmptyMutation, EmptySubscription>;

pub fn api_router() -> Router {
    let schema = Schema::new(Query, EmptyMutation::new(), EmptySubscription::new());

    let app = Router::new()
        .route("/graphql", post(graphql::<Arc<Schema>>))
        .route("/graphql", get(playground("/graphql", None)))
        .route("/graphiql", get(graphiql("/graphql", None)))
        .layer(Extension(Arc::new(schema)));
    app
}
