use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    response::Redirect,
    routing, Router,
};
use hyper::StatusCode;
use juniper::ID;
use serde::Deserialize;
use tracing::error;
use url::Url;

use crate::schema::{
    github_repository_provider::GithubRepositoryProviderService, setting::SettingService,
};

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
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

#[derive(Deserialize)]
struct CallbackParams {
    state: ID,
    code: String,
}

#[derive(Clone)]
struct IntegrationState {
    pub settings: Arc<dyn SettingService>,
    pub github_repository_provider: Arc<dyn GithubRepositoryProviderService>,
}

pub fn routes(
    settings: Arc<dyn SettingService>,
    github_repository_provider: Arc<dyn GithubRepositoryProviderService>,
) -> Router {
    let state = IntegrationState {
        settings,
        github_repository_provider,
    };
    Router::new()
        .route("/connect/:id", routing::get(connect))
        .route("/callback", routing::get(callback))
        .with_state(state)
}

fn get_authorize_url(client_id: &str, redirect_uri: &str, id: &ID) -> Result<Url> {
    Ok(Url::parse_with_params(
        "https://github.com/login/oauth/authorize",
        &[
            ("client_id", client_id),
            ("response_type", "code"),
            ("scope", "repo"),
            (
                "redirect_uri",
                &format!("{redirect_uri}/integrations/github/callback"),
            ),
            ("state", &id.to_string()),
        ],
    )?)
}

async fn exchange_access_token(
    state: &IntegrationState,
    params: &CallbackParams,
) -> Result<GithubOAuthResponse> {
    let client = reqwest::Client::new();

    let client_id = state
        .github_repository_provider
        .get_github_repository_provider(params.state.clone())
        .await?
        .application_id;

    let secret = state
        .github_repository_provider
        .read_github_repository_provider_secret(params.state.clone())
        .await?;

    Ok(client
        .post("https://github.com/login/oauth/access_token")
        .header(reqwest::header::ACCEPT, "application/json")
        .form(&[
            ("client_id", &client_id),
            ("client_secret", &secret),
            ("code", &params.code),
        ])
        .send()
        .await?
        .json()
        .await?)
}

async fn callback(
    State(state): State<IntegrationState>,
    Query(params): Query<CallbackParams>,
) -> Result<Redirect, StatusCode> {
    let response = match exchange_access_token(&state, &params).await {
        Ok(response) => response,
        Err(e) => {
            error!("Failed to exchange access token: {e}");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    if let Err(e) = state
        .github_repository_provider
        .update_github_repository_provider_access_token(params.state, response.access_token)
        .await
    {
        error!("Failed to update github access token: {e}");
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    Ok(Redirect::temporary("/"))
}

async fn connect(
    State(state): State<IntegrationState>,
    Path(id): Path<ID>,
) -> Result<Redirect, StatusCode> {
    let network_setting = match state.settings.read_network_setting().await {
        Ok(setting) => setting,
        Err(e) => {
            error!("Failed to read network setting: {e}");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    let external_url = network_setting.external_url;
    let provider = match state
        .github_repository_provider
        .get_github_repository_provider(id.clone())
        .await
    {
        Ok(provider) => provider,
        Err(e) => {
            error!("Github repository provider not found: {e}");
            return Err(StatusCode::NOT_FOUND);
        }
    };

    let redirect = match get_authorize_url(&provider.application_id, &external_url, &id) {
        Ok(redirect) => redirect,
        Err(e) => {
            error!("Failed to generate callback URL: {e}");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    Ok(Redirect::temporary(redirect.as_str()))
}
