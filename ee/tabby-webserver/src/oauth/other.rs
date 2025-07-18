use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::Deserialize;
use tabby_schema::auth::{AuthenticationService, OAuthCredential, OAuthProvider};

use super::OAuthClient;
use crate::bail;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OtherOAuthResponse {
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
struct OtherUserEmail {
    email: String,
    primary: bool,
    verified: bool,
    visibility: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OtherUser {
    name: String,
}

pub struct OtherClient {
    client: reqwest::Client,
    auth: Arc<dyn AuthenticationService>,
}

#[derive(Debug, Deserialize)]
pub struct OAuthConfig {
    authorization_endpoint	: String,
    token_endpoint: String,
    userinfo_endpoint	: String,
}

impl OtherClient {
    pub fn new(auth: Arc<dyn AuthenticationService>) -> Self {
        Self {
            client: reqwest::Client::new(),
            auth,
        }
    }

    async fn read_credential(&self) -> Result<OAuthCredential> {
        match self
            .auth
            .read_oauth_credential(OAuthProvider::Other)
            .await?
        {
            Some(credential) => Ok(credential),
            None => bail!("No OAuth credential found"),
        }
    }

    async fn retrieve_oidc_config(&self, config_url: Option<String>) -> Result<OAuthConfig> {
        let provider_url = config_url.unwrap_or_else(|| "".to_owned());

        let resp = self
            .client
            .get(provider_url)
            .send()
            .await?
            .json::<OAuthConfig>()
            .await?;

        Ok(resp)
    }
}

#[async_trait]
impl OAuthClient for OtherClient {
    async fn exchange_code_for_token(&self, code: String) -> Result<String> {
        let credential = self.read_credential().await?;
        let redirect_uri = self.auth.oauth_callback_url(OAuthProvider::Other).await?;
        let params: [(&str, &str); 5] = [
            ("client_id", &credential.client_id),
            ("client_secret", &credential.client_secret),
            ("code", &code),
            ("grant_type", "authorization_code"),
            ("redirect_uri", &redirect_uri),
        ];

        let config_url = credential.provider_url;
        let oidc_config = self.retrieve_oidc_config(config_url).await?;
        let token_endpoint = oidc_config.token_endpoint;

        let token_resp = self
            .client
            .post(token_endpoint)
            .header(reqwest::header::ACCEPT, "application/json")
            .form(&params)
            .send()
            .await?
            .json::<OtherOAuthResponse>()
            .await?;

        if !token_resp.error.is_empty() {
            bail!(
                "Failed to exchange access token: {}",
                token_resp.error_description
            );
        }

        Ok(token_resp.access_token)
    }

    async fn fetch_user_email(&self, access_token: &str) -> Result<String> {
        todo!()
    }

    async fn fetch_user_full_name(&self, access_token: &str) -> Result<String> {
        todo!()
    }

    async fn get_authorization_url(&self) -> Result<String> {
        let credential = self.read_credential().await?;

        let config_url = credential.provider_url;
        let oidc_config = self.retrieve_oidc_config(config_url).await?;
        let authorization_endpoint = &oidc_config.authorization_endpoint;

        let response_type = &"code".to_string();
        // TODO: Use the well known endpoint instead of the hardcoded scopes
        let scope = &"openid email profile".to_string();
        let redirect_uri = &self.auth.oauth_callback_url(OAuthProvider::Other).await?;

        let mut url = reqwest::Url::parse(authorization_endpoint)?;
        let params = vec![
            ("client_id", &credential.client_id),
            ("response_type", response_type),
            ("scope", scope),
            ("redirect_uri", redirect_uri),
        ];
        for (k, v) in params {
            url.query_pairs_mut().append_pair(k, v);
        }

       Ok(url.to_string())
    }
}
