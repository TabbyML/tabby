use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    extract::{Path, State},
    http::Request,
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Response},
    routing, Extension, Json, Router,
};
use cached::{CachedAsync, TimedCache};
use hyper::{header::CONTENT_TYPE, Body, StatusCode};
use juniper::ID;
use juniper_axum::{extract::AuthBearer, graphiql, graphql, playground};
use tabby_common::{
    api::{
        code::CodeSearch,
        event::{ComposedLogger, EventLogger},
        server_setting::ServerSetting,
    },
    config::{RepositoryAccess, RepositoryConfig},
};
use tabby_db::DbConn;
use tokio::sync::Mutex;
use tracing::{error, warn};

use crate::{
    cron, hub, integrations, oauth,
    path::db_file,
    repositories,
    schema::{
        auth::AuthenticationService, create_schema, repository::RepositoryService, Schema,
        ServiceLocator,
    },
    service::create_service_locator,
    service::event_logger::create_event_logger,
    ui,
};

struct ServiceRepositoryAccess {
    service: Arc<dyn RepositoryService>,
    url_cache: Mutex<TimedCache<(), Vec<RepositoryConfig>>>,
}

#[async_trait]
impl RepositoryAccess for ServiceRepositoryAccess {
    async fn list_repositories(&self) -> anyhow::Result<Vec<RepositoryConfig>> {
        let mut cache = self.url_cache.lock().await;
        let repos = cache
            .try_get_or_set_with((), || async {
                let repos = self
                    .service
                    .list_repositories(None, None, None, None)
                    .await?;
                Ok::<_, anyhow::Error>(
                    repos
                        .into_iter()
                        .map(|repo| RepositoryConfig::new_named(repo.name, repo.git_url))
                        .collect(),
                )
            })
            .await?
            .clone();

        Ok(repos)
    }
}

fn new_repository_access(service: Arc<dyn RepositoryService>) -> impl RepositoryAccess {
    ServiceRepositoryAccess {
        service,
        url_cache: Mutex::new(TimedCache::with_lifespan(10 * 60)),
    }
}

pub struct WebserverHandle {
    db: DbConn,
    event_logger: Arc<dyn EventLogger>,
    repository_service: Arc<dyn RepositoryService>,
}

impl WebserverHandle {
    pub async fn new(logger1: impl EventLogger + 'static) -> Self {
        let db = DbConn::new(db_file().as_path())
            .await
            .expect("Must be able to initialize db");
        let repository_service = Arc::new(db.clone());

        let logger2 = create_event_logger(db.clone());
        let logger = Arc::new(ComposedLogger::new(logger1, logger2));
        WebserverHandle {
            db,
            event_logger: logger,
            repository_service,
        }
    }

    pub fn logger(&self) -> Arc<dyn EventLogger + 'static> {
        self.event_logger.clone()
    }

    pub fn repository_access(&self) -> Arc<dyn RepositoryAccess> {
        Arc::new(new_repository_access(self.repository_service.clone()))
    }

    pub async fn attach_webserver(
        &self,
        api: Router,
        ui: Router,
        code: Arc<dyn CodeSearch>,
        is_chat_enabled: bool,
        local_port: u16,
    ) -> (Router, Router) {
        let ctx = create_service_locator(
            self.logger(),
            code,
            self.repository_service.clone(),
            self.db.clone(),
            is_chat_enabled,
        )
        .await;
        cron::run_cron(ctx.auth(), ctx.job(), ctx.worker(), local_port).await;

        let schema = Arc::new(create_schema());

        let api = api
            .route(
                "/v1beta/server_setting",
                routing::get(server_setting).with_state(ctx.clone()),
            )
            // Routes before `distributed_tabby_layer` are protected by authentication middleware for following routes:
            // 1. /v1/*
            // 2. /v1beta/*
            .layer(from_fn_with_state(ctx.clone(), distributed_tabby_layer))
            .route(
                "/graphql",
                routing::post(graphql::<Arc<Schema>, Arc<dyn ServiceLocator>>)
                    .with_state(ctx.clone()),
            )
            .route("/graphql", routing::get(playground("/graphql", None)))
            .layer(Extension(schema))
            .route(
                "/hub",
                routing::get(hub::ws_handler).with_state(ctx.clone()),
            )
            .nest(
                "/repositories",
                repositories::routes(ctx.repository(), ctx.auth()),
            )
            .nest(
                "/integrations/github",
                integrations::github::routes(ctx.setting(), ctx.github_repository_provider()),
            )
            .route(
                "/avatar/:id",
                routing::get(avatar)
                    .with_state(ctx.auth())
                    .layer(from_fn_with_state(ctx.auth(), require_login_middleware)),
            )
            .nest("/oauth", oauth::routes(ctx.auth()));

        let ui = ui.route("/graphiql", routing::get(graphiql("/graphql", None)));

        let ui = ui.fallback(ui::handler);

        (api, ui)
    }
}

pub(crate) async fn require_login_middleware(
    State(auth): State<Arc<dyn AuthenticationService>>,
    AuthBearer(token): AuthBearer,
    request: Request<Body>,
    next: Next<Body>,
) -> axum::response::Response {
    let unauthorized = axum::response::Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .body(Body::empty())
        .unwrap()
        .into_response();

    let Some(token) = token else {
        return unauthorized;
    };

    let Ok(_) = auth.verify_access_token(&token).await else {
        return unauthorized;
    };

    next.run(request).await
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
    let security_setting = match locator.setting().read_security_setting().await {
        Ok(x) => x,
        Err(err) => {
            warn!("Failed to read security setting {}", err);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    Ok(Json(ServerSetting {
        disable_client_side_telemetry: security_setting.disable_client_side_telemetry,
    }))
}

async fn avatar(
    State(state): State<Arc<dyn AuthenticationService>>,
    Path(id): Path<ID>,
) -> Result<Response<Body>, StatusCode> {
    let avatar = state
        .get_user_avatar(&id)
        .await
        .map_err(|e| {
            error!("Failed to retrieve avatar: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .ok_or(StatusCode::NOT_FOUND)?;
    let mut response = Response::new(Body::from(avatar.into_vec()));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, "image/*".parse().unwrap());
    Ok(response)
}
