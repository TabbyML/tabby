use std::sync::Arc;
use anyhow::Result;
use hyper::{Client, Method, Request};
use hyper::client::HttpConnector;
use hyper_rustls::HttpsConnector;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};
use crate::schema::auth::{AuthenticationService, OAuthCredential, OAuthProvider};
use crate::schema::ServiceLocator;

#[derive(Debug, Deserialize)]
struct GithubOAuthResponse {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    scope: String,
    #[serde(default)]
    token_type: String,

    #[serde(default)]
    error: String,
    #[serde(default)]
    error_description: String,
    #[serde(default)]
    error_uri: String,
}

#[derive(Debug, Deserialize)]
struct GithubUserEmail {
    email: String,
    primary: bool,
    verified: bool,
    visibility: String,
}

async fn exchange_access_token(
    code: String,
    credential: OAuthCredential,
    client: Client<HttpsConnector<HttpConnector>>,
) -> Result<GithubOAuthResponse> {
    let req = Request::builder()
        .method(Method::POST)
        .uri("https://github.com/login/oauth/access_token")
        .header(hyper::header::ACCEPT, "application/json")
        .body(hyper::Body::from(format!(
            "client_id={}&client_secret={}&code={}",
            credential.client_id, credential.client_secret, code
        )))?;

    let raw = client.request(req).await?;
    let body = hyper::body::to_bytes(raw.into_body()).await?;
    let resp = serde_json::from_slice::<GithubOAuthResponse>(body.as_ref())?;

    Ok(resp)
}

pub async fn fetch_user_email(
    code: String,
    credential: OAuthCredential,
    client: Client<HttpsConnector<HttpConnector>>,
) -> Result<String> {
    let token_resp = exchange_access_token(code, credential, client.clone()).await?;
    if !token_resp.error.is_empty() {
        return Err(anyhow::anyhow!(
            "Failed to exchange access token: {}",
            token_resp.error_description
        ));
    }

    let req = Request::builder()
        .method(Method::GET)
        .uri("https://api.github.com/user/emails")
        .header(hyper::header::USER_AGENT, "Tabby")
        .header(hyper::header::ACCEPT, "application/vnd.github+json")
        .header(hyper::header::AUTHORIZATION, format!("Bearer {}", token_resp.access_token))
        .header("X-GitHub-Api-Version", "2022-11-28")
        .body(hyper::Body::empty())?;
    let raw = client.request(req).await?;
    let body = hyper::body::to_bytes(raw.into_body()).await?;
    let resp = serde_json::from_slice::<Vec<GithubUserEmail>>(body.as_ref())?;

    if resp.len() == 0 {
        return Err(anyhow::anyhow!("No email address found"));
    }
    for item in &resp {
        if item.primary {
            return Ok(item.email.clone());
        }
    }
    Ok(resp[0].email.clone())
}
