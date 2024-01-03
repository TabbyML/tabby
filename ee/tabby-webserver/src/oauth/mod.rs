use std::sync::Arc;
use axum::{Json, Router, routing};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use hyper::Client;
use hyper::client::HttpConnector;
use hyper_rustls::HttpsConnector;
use serde::Deserialize;
use tracing::error;
use crate::schema::auth::{AuthenticationService, GithubAuthError, GithubAuthResponse, OAuthProvider};
use crate::schema::ServiceLocator;

pub(crate) mod github;

#[derive(Clone)]
struct OAuthState {
    auth: Arc<dyn AuthenticationService>,
    client: Client<HttpsConnector<HttpConnector>>,
}

pub fn routes(auth: Arc<dyn AuthenticationService>) -> Router {
    let https = hyper_rustls::HttpsConnectorBuilder::new()
        .with_native_roots()
        .expect("no native root CA certificates found")
        .https_only()
        .enable_all_versions()
        .build();
    let client: Client<_, hyper::Body> = Client::builder().build(https);
    let state = OAuthState { auth, client };

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
) -> Result<Json<GithubAuthResponse>, StatusCode> {
    match state
        .auth
        .github_auth(param.code, state.client.clone())
        .await
    {
        Ok(resp) => Ok(Json(resp)),
        Err(GithubAuthError::InvalidVerificationCode) => Err(StatusCode::BAD_REQUEST),
        Err(GithubAuthError::CredentialNotActive) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            error!("Failed to authenticate with Github: {:?}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
