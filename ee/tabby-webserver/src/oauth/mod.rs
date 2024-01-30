use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Redirect,
    routing, Json, Router,
};
use serde::Deserialize;
use tracing::error;

use crate::{
    oauth::{github::GithubClient, google::GoogleClient},
    schema::auth::{AuthenticationService, OAuthError, OAuthProvider, OAuthResponse},
};

pub mod github;
pub mod google;

pub enum OAuthClient {
    Github(Arc<GithubClient>),
    Google(Arc<GoogleClient>),
}

#[derive(Clone)]
#[non_exhaustive]
struct OAuthState {
    auth: Arc<dyn AuthenticationService>,
    github_client: Arc<GithubClient>,
    google_client: Arc<GoogleClient>,
}

pub fn routes(auth: Arc<dyn AuthenticationService>) -> Router {
    Router::new()
        .route(
            "/providers",
            routing::get(providers_handler).with_state(auth.clone()),
        )
        .nest("/callback", callback_routes(auth))
}

async fn has_provider(auth: &Arc<dyn AuthenticationService>, x: &OAuthProvider) -> bool {
    auth.read_oauth_credential(x.clone())
        .await
        .is_ok_and(|x| x.is_some())
}

async fn providers_handler(
    state: State<Arc<dyn AuthenticationService>>,
) -> Json<Vec<OAuthProvider>> {
    let candidates = vec![OAuthProvider::Google, OAuthProvider::Github];
    let mut providers = vec![];

    for x in candidates {
        if has_provider(&state.0, &x).await {
            providers.push(x);
        }
    }

    Json(providers)
}

fn callback_routes(auth: Arc<dyn AuthenticationService>) -> Router {
    let state = OAuthState {
        auth,
        github_client: Arc::new(GithubClient::default()),
        google_client: Arc::new(GoogleClient::default()),
    };

    Router::new()
        .route("/github", routing::get(github_oauth_handler))
        .route("/google", routing::get(google_oauth_handler))
        .with_state(state.clone())
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct GithubOAuthQueryParam {
    code: String,
    state: Option<String>,
}

async fn github_oauth_handler(
    State(state): State<OAuthState>,
    Query(param): Query<GithubOAuthQueryParam>,
) -> Result<Redirect, StatusCode> {
    match_auth_result(
        state
            .auth
            .oauth(param.code, OAuthClient::Github(state.github_client.clone()))
            .await,
    )
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct GoogleOAuthQueryParam {
    #[serde(default)]
    code: String,
    #[serde(default)]
    scope: String,
    #[serde(default)]
    error: String,
}

async fn google_oauth_handler(
    State(state): State<OAuthState>,
    Query(param): Query<GoogleOAuthQueryParam>,
) -> Result<Redirect, StatusCode> {
    if !param.error.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    match_auth_result(
        state
            .auth
            .oauth(param.code, OAuthClient::Google(state.google_client.clone()))
            .await,
    )
}

fn match_auth_result(result: Result<OAuthResponse, OAuthError>) -> Result<Redirect, StatusCode> {
    match result {
        Ok(resp) => {
            let uri = format!(
                "/auth/signin?refresh_token={}&access_token={}",
                resp.refresh_token, resp.access_token,
            );
            Ok(Redirect::temporary(&uri))
        }
        Err(OAuthError::InvalidVerificationCode) => Err(StatusCode::BAD_REQUEST),
        Err(OAuthError::CredentialNotActive) => Err(StatusCode::NOT_FOUND),
        Err(OAuthError::UserNotInvited) => Err(StatusCode::UNAUTHORIZED),
        Err(e) => {
            error!("Failed to authenticate with Google: {:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
