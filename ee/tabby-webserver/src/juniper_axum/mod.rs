pub mod extract;
pub mod response;

use std::future;

use axum::{
    extract::Extension,
    response::{Html, IntoResponse},
};
use juniper_graphql_ws::Schema;

use self::{extract::JuniperRequest, response::JuniperResponse};

/// [`Handler`], which handles a [`JuniperRequest`] with the specified [`Schema`], by [`extract`]ing
/// it from [`Extension`]s and initializing its fresh [`Schema::Context`] as a [`Default`] one.
///
/// > __NOTE__: This is a ready-to-go default [`Handler`] for serving GraphQL requests. If you need
/// >           to customize it (for example, extract [`Schema::Context`] from [`Extension`]s
/// >           instead initializing a [`Default`] one), create your own [`Handler`] accepting a
/// >           [`JuniperRequest`] (see its documentation for examples).
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
///
/// use axum::{routing::post, Extension, Json, Router};
/// use juniper::{
///     RootNode, EmptySubscription, EmptyMutation, graphql_object,
/// };
/// use juniper_axum::graphql;
///
/// #[derive(Clone, Copy, Debug, Default)]
/// pub struct Context;
///
/// impl juniper::Context for Context {}
///
/// #[derive(Clone, Copy, Debug)]
/// pub struct Query;
///
/// #[graphql_object(context = Context)]
/// impl Query {
///     fn add(a: i32, b: i32) -> i32 {
///         a + b
///     }
/// }
///
/// type Schema = RootNode<'static, Query, EmptyMutation<Context>, EmptySubscription<Context>>;
///
/// let schema = Schema::new(
///    Query,
///    EmptyMutation::<Context>::new(),
///    EmptySubscription::<Context>::new()
/// );
///
/// let app: Router = Router::new()
///     .route("/graphql", post(graphql::<Arc<Schema>>))
///     .layer(Extension(Arc::new(schema)));
/// ```
///
/// [`extract`]: axum::extract
/// [`Handler`]: axum::handler::Handler
#[cfg_attr(text, axum::debug_handler)]
pub async fn graphql<S>(
    Extension(schema): Extension<S>,
    JuniperRequest(req): JuniperRequest<S::ScalarValue>,
) -> impl IntoResponse
where
    S: Schema, // TODO: Refactor in the way we don't depend on `juniper_graphql_ws::Schema` here.
    S::Context: Default,
{
    JuniperResponse(
        req.execute(schema.root_node(), &S::Context::default())
            .await,
    )
    .into_response()
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
