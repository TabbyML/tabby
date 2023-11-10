mod juniper_axum;
mod registry;
mod schema;

use std::sync::Arc;

use axum::{
    routing::{get, post},
    Extension, Router,
};
use juniper_axum::{graphiql, graphql, playground};

pub fn api_router() -> Router {
    let schema = schema::new();

    let app = Router::new()
        .route("/graphql", post(graphql::<Arc<schema::Schema>>))
        .route("/graphql", get(playground("/graphql", None)))
        .route("/graphiql", get(graphiql("/graphql", None)))
        .layer(Extension(Arc::new(schema)));
    app
}
