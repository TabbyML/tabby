mod handler;
mod hub;
mod path;
mod repositories;
mod schema;
mod service;
mod ui;

pub mod public {
    pub use super::{
        handler::attach_webserver,
        /* used by tabby workers (consumer of /hub api) */
        hub::api::{create_client, HubClient, RegisterWorkerRequest, WorkerKind},
        /* used by examples/update-schema.rs */ schema::create_schema,
    };
}

use std::sync::Arc;

use axum::{
    extract::State,
    http::Request,
    middleware::{from_fn_with_state, Next},
    routing, Extension, Router,
};
use hyper::Body;
use juniper_axum::{graphiql, graphql, playground};
use schema::{Schema, ServiceLocator};
use service::create_service_locator;
use tabby_common::api::{code::CodeSearch, event::RawEventLogger};

pub async fn attach_webserver(
    api: Router,
    ui: Router,
    logger: Arc<dyn RawEventLogger>,
    code: Arc<dyn CodeSearch>,
) -> (Router, Router) {
    let ctx = create_service_locator(logger, code).await;
    let schema = Arc::new(schema::create_schema());

    let api = api
        .layer(from_fn_with_state(ctx.clone(), distributed_tabby_layer))
        .route(
            "/graphql",
            routing::post(graphql::<Arc<Schema>, Arc<dyn ServiceLocator>>).with_state(ctx.clone()),
        )
        .route("/graphql", routing::get(playground("/graphql", None)))
        .layer(Extension(schema))
        .route(
            "/hub",
            routing::get(hub::ws_handler).with_state(ctx.clone()),
        )
        .nest("/repositories", repositories::routes(ctx.clone()));

    let ui = ui
        .route("/graphiql", routing::get(graphiql("/graphql", None)))
        .fallback(ui::handler);

    (api, ui)
}

async fn distributed_tabby_layer(
    State(ws): State<Arc<dyn ServiceLocator>>,
    request: Request<Body>,
    next: Next<Body>,
) -> axum::response::Response {
    ws.worker().dispatch_request(request, next).await
}
