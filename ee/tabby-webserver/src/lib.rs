mod schema;
mod ui;
mod webserver;
mod worker;

use std::sync::Arc;

use axum::{
    extract::State,
    http::Request,
    middleware::{from_fn_with_state, Next},
    routing, Extension, Router,
};
use hyper::Body;
use juniper::EmptySubscription;
use juniper_axum::{graphiql, graphql, playground};
use schema::{Mutation, Query, Schema};
use webserver::Webserver;

pub fn attach_webserver(router: Router) -> Router {
    let ws = Arc::new(Webserver::default());
    let schema = Arc::new(Schema::new(Query, Mutation, EmptySubscription::new()));

    let app = Router::new()
        .route("/graphql", routing::get(playground("/graphql", None)))
        .route("/graphiql", routing::get(graphiql("/graphql", None)))
        .route(
            "/graphql",
            routing::post(graphql::<Arc<Schema>, Arc<Webserver>>).with_state(ws.clone()),
        )
        .layer(Extension(schema));

    router
        .merge(app)
        .fallback(ui::handler)
        .layer(from_fn_with_state(ws, distributed_tabby_layer))
}

async fn distributed_tabby_layer(
    State(ws): State<Arc<Webserver>>,
    request: Request<Body>,
    next: Next<Body>,
) -> axum::response::Response {
    ws.dispatch_request(request, next).await
}
