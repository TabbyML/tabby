mod juniper_axum;
mod registry;
mod schema;

use std::sync::Arc;

use axum::{
    routing::{get, post},
    Extension, Router,
};
use juniper_axum::{graphiql, graphql, playground};
use schema::Context;
use tabby_common::api::code::SearchResponse;

pub fn api_router() -> Router {
    let context = Context::default();
    let schema = schema::new();

    let app = Router::new()
        .route(
            "/graphql",
            post(graphql::<Arc<schema::Schema>>).with_state(Arc::new(context)),
        )
        .route("/graphql", get(playground("/graphql", None)))
        .route("/graphiql", get(graphiql("/graphql", None)))
        .layer(Extension(Arc::new(schema)));
    app
}

#[tarpc::service]
trait CodeSearchService {
    async fn search(q: String, limit: usize, offset: usize) -> SearchResponse;

    async fn search_in_language(
        language: String,
        tokens: Vec<String>,
        limit: usize,
        offset: usize,
    ) -> SearchResponse;
}
