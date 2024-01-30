use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;
use tabby_db::GoogleOAuthCredentialDAO;

use crate::schema::auth::{AuthenticationService, OAuthCredential, OAuthProvider};

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GoogleOAuthResponse {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    expires_in: i32,
    #[serde(default)]
    token_type: String,
    #[serde(default)]
    scope: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GoogleOAuthError {
    code: i32,
    message: String,
    status: String,
}

#[derive(Debug, Deserialize)]
struct GoogleUserEmail {
    #[serde(default)]
    email: String,
    error: Option<GoogleOAuthError>,
}

#[derive(Default)]
pub struct GoogleClient {
    client: reqwest::Client,
}

impl GoogleClient {
    pub async fn fetch_user_email(
        &self,
        code: String,
        credential: OAuthCredential,
    ) -> Result<String> {
        let token_resp = self.exchange_access_token(code, credential).await?;
        if token_resp.access_token.is_empty() {
            return Err(anyhow::anyhow!("Empty access token from Google OAuth"));
        }

        let resp = self
            .client
            .get("https://www.googleapis.com/oauth2/v2/userinfo?alt=json&fields=email")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", token_resp.access_token),
            )
            .send()
            .await?
            .json::<GoogleUserEmail>()
            .await?;

        if let Some(err) = resp.error {
            return Err(anyhow::anyhow!(err.message));
        }
        Ok(resp.email)
    }

    async fn exchange_access_token(
        &self,
        code: String,
        credential: OAuthCredential,
    ) -> Result<GoogleOAuthResponse> {
        let Some(client_secret) = credential.client_secret else {
            return Err(anyhow::anyhow!("Missing client secret"))
        };

        let Some(redirect_uri) = credential.redirect_uri else {
            return Err(anyhow::anyhow!("Missing redirect uri"))
        };

        let params = [
            ("client_id", credential.client_id.as_str()),
            ("client_secret", client_secret.as_str()),
            ("code", code.as_str()),
            ("grant_type", "authorization_code"),
            ("redirect_uri", redirect_uri.as_str()),
        ];

        let resp = self
            .client
            .post("https://oauth2.googleapis.com/token")
            .form(&params)
            .send()
            .await?
            .json::<GoogleOAuthResponse>()
            .await?;

        Ok(resp)
    }

    pub async fn get_authorization_url(&self, credential: OAuthCredential) -> Result<String> {
        let mut url = reqwest::Url::parse("https://accounts.google.com/o/oauth2/v2/auth")?;
        let redirect_uri = credential
            .redirect_uri
            .context("Google OAuth requires redirect_uri")?;
        let params = vec![
            ("client_id", credential.client_id.as_str()),
            ("redirect_uri", redirect_uri.as_str()),
            ("response_type", "code"),
            ("scope", "https://www.googleapis.com/auth/userinfo.email"),
            ("access_type", "offline"),
        ];
        for (k, v) in params {
            url.query_pairs_mut().append_pair(k, v);
        }
        Ok(url.to_string())
    }
}
