use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;
use tabby_db::GithubOAuthCredentialDAO;
use tokio_tungstenite::tungstenite::client;
use crate::schema::auth::{AuthenticationService, OAuthCredential, OAuthProvider};

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

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GithubUserEmail {
    email: String,
    primary: bool,
    verified: bool,
    visibility: Option<String>,
}

#[derive(Default)]
pub struct GithubClient {
    client: reqwest::Client,
}

impl GithubClient {
    pub async fn fetch_user_email(
        &self,
        code: String,
        credential: OAuthCredential,
    ) -> Result<String> {
        let token_resp = self.exchange_access_token(code, credential).await?;
        if !token_resp.error.is_empty() {
            return Err(anyhow::anyhow!(
                "Failed to exchange access token: {}",
                token_resp.error_description
            ));
        }

        let resp = self
            .client
            .get("https://api.github.com/user/emails")
            .header(reqwest::header::USER_AGENT, "Tabby")
            .header(reqwest::header::ACCEPT, "application/vnd.github+json")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", token_resp.access_token),
            )
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await?;

        let emails = resp.json::<Vec<GithubUserEmail>>()
            .await?;

        for item in &emails {
            if item.primary {
                return Ok(item.email.clone());
            }
        }

        return Err(anyhow::anyhow!("No primary email address found"));
    }

    async fn exchange_access_token(
        &self,
        code: String,
        credential: OAuthCredential,
    ) -> Result<GithubOAuthResponse> {
        let Some(client_secret) = credential.client_secret else {
            return Err(anyhow::anyhow!("Missing client secret"))
        };

        let params = [
            ("client_id", credential.client_id.as_str()),
            ("client_secret", client_secret.as_str()),
            ("code", code.as_str()),
        ];
        let resp = self
            .client
            .post("https://github.com/login/oauth/access_token")
            .header(reqwest::header::ACCEPT, "application/json")
            .form(&params)
            .send()
            .await?
            .json::<GithubOAuthResponse>()
            .await?;

        Ok(resp)
    }

    pub async fn get_authorization_url(&self, credential: OAuthCredential) -> Result<String> {
        let mut url = reqwest::Url::parse("https://github.com/login/oauth/authorize")?;
        let params = vec![
            ("client_id", credential.client_id.as_str()),
            ("response_type", "code"),
            ("scope", "read:user user:email"),
        ];
        for (k, v) in params {
            url.query_pairs_mut().append_pair(k, v);
        }
        Ok(url.to_string())
    }
}
