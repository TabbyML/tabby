use anyhow::Result;
use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    response::Redirect,
    routing, Router,
};
use hyper::StatusCode;
use juniper::ID;
use serde::Deserialize;

use crate::{
    oauth::github::GithubOAuthResponse,
    schema::{
        github_repository_provider::GithubRepositoryProviderService, setting::SettingService,
    },
};

#[derive(Clone)]
pub struct OAuthState {
    pub settings: Arc<dyn SettingService>,
    pub github_repository_provider: Arc<dyn GithubRepositoryProviderService>,
}

pub fn routes(state: OAuthState) -> Router {
    Router::new()
        .route("/github/login/:id", routing::get(login))
        .route("/github/callback", routing::get(callback))
        .with_state(state)
}

fn github_redirect_url(client_id: &str, redirect_uri: &str, id: &ID) -> String {
    format!("https://github.com/login/oauth/authorize?client_id={client_id}&response_type=code&scope=repo&redirect_uri={redirect_uri}/repositories/oauth/github/callback&state={id}")
}

#[derive(Deserialize)]
struct CallbackParams {
    state: ID,
    code: String,
}

macro_rules! log_error {
    ($val:expr) => {
        $val.map_err(|e| {
            tracing::error!("{e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })
    };
}

async fn exchange_access_token(
    state: &OAuthState,
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
    State(state): State<OAuthState>,
    Query(params): Query<CallbackParams>,
) -> Result<Redirect, StatusCode> {
    let network_setting = log_error!(state.settings.read_network_setting().await)?;
    let external_url = network_setting.external_url;

    let response = log_error!(exchange_access_token(&state, &params).await)?;
    dbg!(&response);
    log_error!(
        state
            .github_repository_provider
            .set_github_repository_provider_token(params.state, response.access_token)
            .await
    )?;

    Ok(Redirect::permanent(&external_url))
}

async fn login(
    State(state): State<OAuthState>,
    Path(id): Path<ID>,
) -> Result<Redirect, StatusCode> {
    let network_setting = log_error!(state.settings.read_network_setting().await)?;
    let external_url = network_setting.external_url;
    let client_id = log_error!(
        state
            .github_repository_provider
            .get_github_repository_provider(id.clone())
            .await
    )?
    .application_id;
    Ok(Redirect::temporary(&github_redirect_url(
        &client_id,
        &external_url,
        &id,
    )))
}
