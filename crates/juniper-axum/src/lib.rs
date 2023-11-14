pub mod extract;
pub mod response;

use std::{future, sync::Arc};

use axum::{
    extract::{Extension, State},
    response::{Html, IntoResponse},
};
use juniper_graphql_ws::Schema;

use self::{extract::JuniperRequest, response::JuniperResponse};

#[cfg_attr(text, axum::debug_handler)]
pub async fn graphql<S>(
    State(state): State<Arc<S::Context>>,
    Extension(schema): Extension<S>,
    JuniperRequest(req): JuniperRequest<S::ScalarValue>,
) -> impl IntoResponse
where
    S: Schema, // TODO: Refactor in the way we don't depend on `juniper_graphql_ws::Schema` here.
{
    JuniperResponse(req.execute(schema.root_node(), &state).await).into_response()
}

/// Creates a [`Handler`] that replies with an HTML page containing [GraphiQL].
///
/// This does not handle routing, so you can mount it on any endpoint.
///
/// # Example
///
/// ```rust
/// use axum::{routing::get, Router};
/// use juniper_axum::graphiql;
///
/// let app: Router = Router::new()
///     .route("/", get(graphiql("/graphql", "/subscriptions")));
/// ```
///
/// [`Handler`]: axum::handler::Handler
/// [GraphiQL]: https://github.com/graphql/graphiql
pub fn graphiql<'a>(
    graphql_endpoint_url: &str,
    subscriptions_endpoint_url: impl Into<Option<&'a str>>,
) -> impl FnOnce() -> future::Ready<Html<String>> + Clone + Send {
    let html = Html(juniper::http::graphiql::graphiql_source(
        graphql_endpoint_url,
        subscriptions_endpoint_url.into(),
    ));

    || future::ready(html)
}

/// Creates a [`Handler`] that replies with an HTML page containing [GraphQL Playground].
///
/// This does not handle routing, so you can mount it on any endpoint.
///
/// # Example
///
/// ```rust
/// use axum::{routing::get, Router};
/// use juniper_axum::playground;
///
/// let app: Router = Router::new()
///     .route("/", get(playground("/graphql", "/subscriptions")));
/// ```
///
/// [`Handler`]: axum::handler::Handler
/// [GraphQL Playground]: https://github.com/prisma/graphql-playground
pub fn playground<'a>(
    graphql_endpoint_url: &str,
    subscriptions_endpoint_url: impl Into<Option<&'a str>>,
) -> impl FnOnce() -> future::Ready<Html<String>> + Clone + Send {
    let html = Html(juniper::http::playground::playground_source(
        graphql_endpoint_url,
        subscriptions_endpoint_url.into(),
    ));

    || future::ready(html)
}
