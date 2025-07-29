use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use tabby_schema::auth::{AuthenticationService, OAuthCredential, OAuthProvider};

use super::OAuthClient;
use crate::bail;

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

#[derive(Debug, Deserialize)]
struct GoogleUserName {
    #[serde(default)]
    name: String,
    error: Option<GoogleOAuthError>,
}

pub struct GoogleClient {
    client: reqwest::Client,
    auth: Arc<dyn AuthenticationService>,
}

impl GoogleClient {
    pub fn new(auth: Arc<dyn AuthenticationService>) -> Self {
        Self {
            client: reqwest::Client::new(),
            auth,
        }
    }

    async fn read_credential(&self) -> Result<OAuthCredential> {
        match self
            .auth
            .read_oauth_credential(OAuthProvider::Google)
            .await?
        {
            Some(credential) => Ok(credential),
            None => bail!("No Google OAuth credential found"),
        }
    }

    async fn exchange_access_token(
        &self,
        code: String,
        credential: OAuthCredential,
        redirect_uri: String,
    ) -> Result<GoogleOAuthResponse> {
        let client_secret = credential.client_secret;

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
}

#[async_trait]
impl OAuthClient for GoogleClient {
    async fn exchange_code_for_token(&self, code: String) -> Result<String> {
        let credential = self.read_credential().await?;
        let redirect_uri = self.auth.oauth_callback_url(OAuthProvider::Google).await?;
        let token_resp = self
            .exchange_access_token(code, credential, redirect_uri)
            .await?;
        if token_resp.access_token.is_empty() {
            bail!("Empty access token from Google OAuth");
        }

        Ok(token_resp.access_token)
    }

    async fn fetch_user_email(&self, access_token: &str) -> Result<String> {
        let resp = self
            .client
            .get("https://www.googleapis.com/oauth2/v2/userinfo?alt=json&fields=email")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {access_token}"),
            )
            .send()
            .await?
            .json::<GoogleUserEmail>()
            .await?;

        if let Some(err) = resp.error {
            bail!(err.message);
        }
        Ok(resp.email)
    }

    async fn fetch_user_full_name(&self, access_token: &str) -> Result<String> {
        let resp = self
            .client
            .get("https://www.googleapis.com/oauth2/v2/userinfo?alt=json&fields=name")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {access_token}"),
            )
            .send()
            .await?
            .json::<GoogleUserName>()
            .await?;

        if let Some(err) = resp.error {
            bail!(err.message);
        }
        Ok(resp.name)
    }

    async fn get_authorization_url(&self) -> Result<String> {
        let credential = self.read_credential().await?;
        let redirect_uri = self.auth.oauth_callback_url(OAuthProvider::Google).await?;
        create_authorization_url(&credential.client_id, &redirect_uri)
    }
}

fn create_authorization_url(client_id: &str, redirect_uri: &str) -> Result<String> {
    let mut url = reqwest::Url::parse("https://accounts.google.com/o/oauth2/v2/auth")?;
    let params = vec![
        ("client_id", client_id),
        ("redirect_uri", redirect_uri),
        ("response_type", "code"),
        ("scope", "https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile"),
        ("access_type", "offline"),
    ];
    for (k, v) in params {
        url.query_pairs_mut().append_pair(k, v);
    }
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::create_authorization_url;

    #[test]
    fn test_create_authorization_url() {
        let url = create_authorization_url("client_id", "localhost").unwrap();
        assert_eq!(url, "https://accounts.google.com/o/oauth2/v2/auth?client_id=client_id\
        &redirect_uri=localhost\
        &response_type=code\
        &scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.email+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.profile\
        &access_type=offline");
    }
}
