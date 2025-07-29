use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Redirect,
    routing, Json, Router,
};
use serde::Deserialize;
use strum::IntoEnumIterator;
use tabby_schema::auth::{AuthenticationService, OAuthError, OAuthProvider, OAuthResponse};
use tracing::error;

use crate::oauth;

type OAuthState = Arc<dyn AuthenticationService>;

pub fn routes(state: Arc<dyn AuthenticationService>) -> Router {
    Router::new()
        .route("/signin", routing::get(signin_handler))
        .route("/providers", routing::get(providers_handler))
        .route("/callback/github", routing::get(github_oauth_handler))
        .route("/callback/google", routing::get(google_oauth_handler))
        .route("/callback/gitlab", routing::get(gitlab_oauth_handler))
        .with_state(state)
}

#[derive(Deserialize)]
struct SigninQueryParams {
    provider: OAuthProvider,
}

async fn signin_handler(
    State(state): State<OAuthState>,
    Query(params): Query<SigninQueryParams>,
) -> Result<Redirect, StatusCode> {
    let redirect_uri = oauth::new_oauth_client(params.provider, state)
        .get_authorization_url()
        .await;

    match redirect_uri {
        Ok(uri) => Ok(Redirect::temporary(&uri)),
        Err(e) => {
            error!("Failed to get authorization URL: {:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn has_provider(auth: &Arc<dyn AuthenticationService>, x: &OAuthProvider) -> bool {
    auth.read_oauth_credential(x.clone())
        .await
        .is_ok_and(|x| x.is_some())
}

async fn providers_handler(state: State<OAuthState>) -> Json<Vec<OAuthProvider>> {
    let mut providers = vec![];

    for x in OAuthProvider::iter() {
        if has_provider(&state, &x).await {
            providers.push(x);
        }
    }

    Json(providers)
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
) -> Redirect {
    match_auth_result(
        OAuthProvider::Github,
        state.oauth(param.code, OAuthProvider::Github).await,
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
) -> Redirect {
    if !param.error.is_empty() {
        return make_error_redirect(OAuthProvider::Google, param.error);
    }
    match_auth_result(
        OAuthProvider::Google,
        state.oauth(param.code, OAuthProvider::Google).await,
    )
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct GitlabOAuthQueryParam {
    code: String,
    state: Option<String>,
}

async fn gitlab_oauth_handler(
    State(state): State<OAuthState>,
    Query(param): Query<GitlabOAuthQueryParam>,
) -> Redirect {
    match_auth_result(
        OAuthProvider::Gitlab,
        state.oauth(param.code, OAuthProvider::Gitlab).await,
    )
}

fn match_auth_result(
    provider: OAuthProvider,
    result: Result<OAuthResponse, OAuthError>,
) -> Redirect {
    match result {
        Ok(resp) => {
            let uri = format!(
                "/auth/signin?refresh_token={}&access_token={}",
                resp.refresh_token, resp.access_token,
            );
            Redirect::temporary(&uri)
        }
        Err(err) => match err {
            OAuthError::UserNotInvited
            | OAuthError::UserDisabled
            | OAuthError::InsufficientSeats
            | OAuthError::Unknown => make_error_redirect(provider, err.to_string()),
            OAuthError::Other(e) => {
                error!("Failed to authenticate: {:?}", e);
                make_error_redirect(provider, OAuthError::Unknown.to_string())
            }
        },
    }
}

fn make_error_redirect(provider: OAuthProvider, message: String) -> Redirect {
    let query = querystring::stringify(vec![
        ("error_message", urlencoding::encode(&message).as_ref()),
        (
            "provider",
            serde_json::to_string(&provider).unwrap().as_str(),
        ),
    ]);
    let uri = format!("/auth/signin?{query}");
    Redirect::temporary(&uri)
}
