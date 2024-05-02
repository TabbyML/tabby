use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::Request,
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Response},
    routing, Extension, Json, Router,
};
use hyper::{header::CONTENT_TYPE, Body, StatusCode};
use juniper::ID;
use juniper_axum::{graphiql, playground};
use tabby_common::{
    api::{
        code::CodeSearch,
        event::{ComposedLogger, EventLogger},
        server_setting::ServerSetting,
    },
    config::RepositoryAccess,
};
use tabby_db::DbConn;
use tracing::{error, warn};

use crate::{
    axum::{extract::AuthBearer, graphql},
    hub::{self, HubState},
    oauth,
    path::db_file,
    repositories,
    schema::{
        auth::AuthenticationService, create_schema, repository::RepositoryService, Schema,
        ServiceLocator,
    },
    service::{
        background_job, create_service_locator, event_logger::create_event_logger, repository,
    },
    ui,
};

pub struct WebserverHandle {
    db: DbConn,
    logger: Arc<dyn EventLogger>,
    repository: Arc<dyn RepositoryService>,
}

impl WebserverHandle {
    pub async fn new(logger1: impl EventLogger + 'static, local_port: u16) -> Self {
        let db = DbConn::new(db_file().as_path())
            .await
            .expect("Must be able to initialize db");
        db.finalize_stale_job_runs().await.expect("Must be able to finalize stale job runs");

        let background_job = background_job::create(db.clone(), local_port).await;
        let repository = repository::create(db.clone(), background_job);

        let logger2 = create_event_logger(db.clone());
        let logger = Arc::new(ComposedLogger::new(logger1, logger2));
        WebserverHandle {
            db,
            logger,
            repository,
        }
    }

    pub fn logger(&self) -> Arc<dyn EventLogger + 'static> {
        self.logger.clone()
    }

    pub fn repository_access(&self) -> Arc<dyn RepositoryAccess> {
        self.repository.clone().access()
    }

    pub async fn attach_webserver(
        &self,
        api: Router,
        ui: Router,
        code: Arc<dyn CodeSearch>,
        is_chat_enabled: bool,
    ) -> (Router, Router) {
        let ctx = create_service_locator(
            self.logger(),
            code,
            self.repository.clone(),
            self.db.clone(),
            is_chat_enabled,
        )
        .await;

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
                routing::get(hub::ws_handler).with_state(
                    HubState::new(ctx.clone(), self.repository_access().clone()).into(),
                ),
            )
            .nest(
                "/repositories",
                repositories::routes(ctx.repository(), ctx.auth()),
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
