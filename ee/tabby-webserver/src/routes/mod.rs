mod hub;
mod oauth;
mod repositories;
mod ui;

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path, State},
    http::{Request, StatusCode},
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Response},
    routing, Extension, Json, Router,
};
use juniper::ID;
use juniper_axum::{graphiql, playground};
use tabby_common::api::server_setting::ServerSetting;
use tabby_schema::{auth::AuthenticationService, create_schema, Schema, ServiceLocator};
use tracing::{error, warn};

use self::hub::HubState;
use crate::{
    axum::{extract::AuthBearer, graphql, FromAuth},
    jwt::{generate_jwt_payload, validate_jwt},
    service::answer::AnswerService,
};

pub fn create(
    ctx: Arc<dyn ServiceLocator>,
    api: Router,
    ui: Router,
    _answer: Option<Arc<AnswerService>>,
) -> (Router, Router) {
    let schema = Arc::new(create_schema());

    let api = api.route(
        "/v1beta/server_setting",
        routing::get(server_setting).with_state(ctx.clone()),
    );

    let api = api
        // Routes before `distributed_tabby_layer` are protected by authentication middleware for following routes:
        // 1. /v1/*
        // 2. /v1beta/*
        .layer(from_fn_with_state(ctx.clone(), distributed_tabby_layer))
        .route(
            "/graphql",
            routing::post(graphql::<Arc<Schema>, Arc<dyn ServiceLocator>>).with_state(ctx.clone()),
        )
        .route(
            "/subscriptions",
            routing::get(crate::axum::subscriptions::<Arc<Schema>, Arc<dyn ServiceLocator>>)
                .with_state(ctx.clone()),
        )
        .route(
            "/graphql",
            routing::get(playground("/graphql", "/subscriptions")),
        )
        .layer(Extension(schema))
        .route(
            "/hub",
            routing::get(hub::ws_handler).with_state(HubState::new(ctx.clone()).into()),
        )
        .nest(
            "/repositories",
            repositories::routes(ctx.repository(), ctx.auth()),
        )
        .route("/avatar/:id", routing::get(avatar).with_state(ctx.auth()))
        .nest("/oauth", oauth::routes(ctx.auth()));

    let ui = ui.route("/graphiql", routing::get(graphiql("/graphql", None)));

    let ui = ui.fallback(ui::handler);

    (api, ui)
}

pub(crate) async fn require_login_middleware(
    State(auth): State<Arc<dyn AuthenticationService>>,
    AuthBearer(token): AuthBearer,
    mut request: Request<Body>,
    next: Next,
) -> axum::response::Response {
    let unauthorized = axum::response::Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .body(Body::empty())
        .unwrap()
        .into_response();

    let Some(token) = token else {
        return unauthorized;
    };

    let Ok(jwt) = auth.verify_access_token(&token).await else {
        return unauthorized;
    };

    let Ok(user) = auth.get_user(&jwt.sub).await else {
        return unauthorized;
    };

    request.extensions_mut().insert(user.policy);

    next.run(request).await
}

async fn distributed_tabby_layer(
    State(ws): State<Arc<dyn ServiceLocator>>,
    request: Request<Body>,
    next: Next,
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
        .insert("Content-Type", "image/*".parse().unwrap());
    Ok(response)
}

#[async_trait::async_trait]
impl FromAuth<Arc<dyn ServiceLocator>> for tabby_schema::Context {
    async fn build(locator: Arc<dyn ServiceLocator>, token: Option<String>) -> Self {
        let claims = if let Some(token) = token {
            let mut claims = validate_jwt(&token).ok();

            if claims.is_none() {
                claims = locator
                    .auth()
                    .verify_auth_token(&token)
                    .await
                    .ok()
                    .map(|id| generate_jwt_payload(id, true));
            }

            claims
        } else {
            None
        };

        Self { claims, locator }
    }
}
