use std::{process::Stdio, sync::Arc};

use axum::{
    extract::State,
    http::Request,
    middleware::{from_fn_with_state, Next},
    routing, Extension, Router,
};
use hyper::Body;
use juniper_axum::{graphiql, graphql, playground};
use tabby_common::{
    api::{code::CodeSearch, event::RawEventLogger},
    config::Config,
};

use crate::{
    hub, oauth,
    repositories::{self, RepositoryCache},
    schema::{create_schema, Schema, ServiceLocator},
    service::create_service_locator,
    ui,
};

pub async fn attach_webserver(
    api: Router,
    ui: Router,
    logger: Arc<dyn RawEventLogger>,
    code: Arc<dyn CodeSearch>,
    config: &Config,
    local_port: u16,
) -> (Router, Router) {
    let repository_cache = Arc::new(RepositoryCache::new_initialized(
        config.repositories.clone(),
    ));
    repository_cache.start_reload_job().await;
    let ctx = create_service_locator(logger, code).await;
    let schema = Arc::new(create_schema());
    let rs = Arc::new(repository_cache);

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
        .nest(
            "/repositories",
            repositories::routes(rs.clone(), ctx.auth()),
        )
        .nest("/oauth", oauth::routes(ctx.auth()));

    let ui = ui
        .route("/graphiql", routing::get(graphiql("/graphql", None)))
        .fallback(ui::handler);

    tokio::spawn(async move {
        loop {
            // Give some time for server being ready.
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            start_scheduler_job(
                local_port,
                ctx.worker().read_registration_token().await.unwrap(),
            )
            .await;
        }
    });
    (api, ui)
}

async fn distributed_tabby_layer(
    State(ws): State<Arc<dyn ServiceLocator>>,
    request: Request<Body>,
    next: Next<Body>,
) -> axum::response::Response {
    ws.worker().dispatch_request(request, next).await
}

async fn start_scheduler_job(local_port: u16, registeration_token: String) {
    let exe = std::env::current_exe().unwrap();
    let mut child = tokio::process::Command::new(exe)
        .arg("scheduler")
        .arg("--url")
        .arg(format!("localhost:{local_port}"))
        .arg("--token")
        .arg(registeration_token)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let _ = child.wait().await;
}
