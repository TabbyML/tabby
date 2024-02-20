use std::sync::Arc;

use axum::{
    extract::State,
    http::Request,
    middleware::{from_fn_with_state, Next},
    routing, Extension, Json, Router,
};
use hyper::{Body, StatusCode};
use juniper_axum::{graphiql, graphql, playground};
use tabby_common::{
    api::{code::CodeSearch, event::RawEventLogger, server_setting::ServerSetting},
    config::Config,
};
use tracing::warn;

use crate::{
    cron, hub, oauth,
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
    is_chat_enabled: bool,
    local_port: u16,
) -> (Router, Router) {
    let ctx = create_service_locator(logger, code, is_chat_enabled).await;
    cron::run_cron(ctx.auth(), ctx.job(), ctx.worker(), local_port).await;

    let repository_cache = Arc::new(RepositoryCache::new_initialized(
        config.repositories.clone(),
    ));
    repository_cache.start_reload_job().await;

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
        .route(
            "/v1beta/server_setting",
            routing::get(server_setting).with_state(ctx.clone()),
        )
        .nest(
            "/repositories",
            repositories::routes(rs.clone(), ctx.auth()),
        )
        .nest("/oauth", oauth::routes(ctx.auth()));

    let ui = ui.route("/graphiql", routing::get(graphiql("/graphql", None)));

    let ui = ui.fallback(ui::handler);

    (api, ui)
}

async fn distributed_tabby_layer(
    State(ws): State<Arc<dyn ServiceLocator>>,
    request: Request<Body>,
    next: Next<Body>,
) -> axum::response::Response {
    ws.worker().dispatch_request(request, next).await
}

async fn server_setting(
    State(locator): State<Arc<dyn ServiceLocator>>,
) -> Result<Json<ServerSetting>, StatusCode> {
    let setting = match locator.setting().read_security_setting().await {
        Ok(x) => x,
        Err(err) => {
            warn!("Failed to read security setting {}", err);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    Ok(Json(ServerSetting {
        disable_client_side_telemetry: setting.disable_client_side_telemetry,
    }))
}
