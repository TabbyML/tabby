use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Redirect,
    routing, Router,
};
use serde::Deserialize;
use tracing::error;

use crate::{
    oauth::github::GithubClient,
    schema::{
        auth::{AuthenticationService, GithubAuthError},
        ServiceLocator,
    },
};

pub mod github;

#[derive(Clone)]
#[non_exhaustive]
struct OAuthState {
    auth: Arc<dyn AuthenticationService>,
    github_client: Arc<GithubClient>,
}

pub fn routes(auth: Arc<dyn AuthenticationService>) -> Router {
    let state = OAuthState {
        auth,
        github_client: Arc::new(GithubClient::new()),
    };

    Router::new()
        .route("/github", routing::get(github_callback))
        .with_state(state)
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct GithubCallbackParam {
    code: String,
    state: Option<String>,
}

async fn github_callback(
    State(state): State<OAuthState>,
    Query(param): Query<GithubCallbackParam>,
) -> Result<Redirect, StatusCode> {
    match state
        .auth
        .github_auth(param.code, state.github_client.clone())
        .await
    {
        Ok(resp) => {
            let uri = format!(
                "/auth/signin?refresh_token={}&access_token={}",
                resp.refresh_token, resp.access_token,
            );
            Ok(Redirect::temporary(&uri))
        }
        Err(GithubAuthError::InvalidVerificationCode) => Err(StatusCode::BAD_REQUEST),
        Err(GithubAuthError::CredentialNotActive) => Err(StatusCode::NOT_FOUND),
        Err(GithubAuthError::UserNotInvited) => Err(StatusCode::UNAUTHORIZED),
        Err(e) => {
            error!("Failed to authenticate with Github: {:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
