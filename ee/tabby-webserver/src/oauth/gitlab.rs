use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use tabby_schema::auth::{AuthenticationService, OAuthCredential, OAuthProvider};

use super::OAuthClient;
use crate::bail;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GitlabOAuthResponse {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    scope: String,
    #[serde(default)]
    token_type: String,

    #[serde(default)]
    expires_in: i32,
    #[serde(default)]
    created_at: u64,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    error_description: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GitlabUserEmail {
    #[serde(default)]
    email: String,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GitlabUserName {
    #[serde(default)]
    name: String,
    error: Option<String>,
}

pub struct GitlabClient {
    client: reqwest::Client,
    auth: Arc<dyn AuthenticationService>,
}

impl GitlabClient {
    pub fn new(auth: Arc<dyn AuthenticationService>) -> Self {
        Self {
            client: reqwest::Client::new(),
            auth,
        }
    }

    async fn read_credential(&self) -> Result<OAuthCredential> {
        match self
            .auth
            .read_oauth_credential(OAuthProvider::Gitlab)
            .await?
        {
            Some(credential) => Ok(credential),
            None => bail!("No Gitlab OAuth credential found"),
        }
    }

    async fn exchange_access_token(
        &self,
        code: String,
        credential: OAuthCredential,
        redirect_uri: String,
    ) -> Result<GitlabOAuthResponse> {
        let params: [(&str, &str); 5] = [
            ("client_id", &credential.client_id),
            ("client_secret", &credential.client_secret),
            ("code", &code),
            ("grant_type", "authorization_code"),
            ("redirect_uri", &redirect_uri),
        ];
        let resp = self
            .client
            .post("https://gitlab.com/oauth/token")
            .header(reqwest::header::ACCEPT, "application/json")
            .form(&params)
            .send()
            .await?
            .json::<GitlabOAuthResponse>()
            .await?;

        Ok(resp)
    }
}

#[async_trait]
impl OAuthClient for GitlabClient {
    async fn exchange_code_for_token(&self, code: String) -> Result<String> {
        let credentials = self.read_credential().await?;
        let redirect_uri = self.auth.oauth_callback_url(OAuthProvider::Gitlab).await?;
        let token_resp = self
            .exchange_access_token(code, credentials, redirect_uri)
            .await?;

        if let Some(err) = token_resp.error {
            bail!(
                "Error while exchanging access token: {err} {}",
                token_resp
                    .error_description
                    .map(|s| format!("({s})"))
                    .unwrap_or_default()
            );
        }

        if token_resp.access_token.is_empty() {
            bail!("Empty access token from Gitlab OAuth");
        }
        Ok(token_resp.access_token)
    }

    async fn fetch_user_email(&self, access_token: &str) -> Result<String> {
        let resp = self
            .client
            .get("https://gitlab.com/api/v4/user")
            .header(reqwest::header::USER_AGENT, "Tabby")
            .header(reqwest::header::ACCEPT, "application/vnd.gitlab+json")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {access_token}"),
            )
            .send()
            .await?;

        let email = resp.json::<GitlabUserEmail>().await?;
        if let Some(error) = email.error {
            bail!("{error}");
        }
        Ok(email.email)
    }

    async fn fetch_user_full_name(&self, access_token: &str) -> Result<String> {
        let resp = self
            .client
            .get("https://gitlab.com/api/v4/user")
            .header(reqwest::header::USER_AGENT, "Tabby")
            .header(reqwest::header::ACCEPT, "application/vnd.gitlab+json")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {access_token}"),
            )
            .send()
            .await?;

        let res = resp.json::<GitlabUserName>().await?;
        if let Some(error) = res.error {
            bail!("{error}");
        }
        Ok(res.name)
    }

    async fn get_authorization_url(&self) -> Result<String> {
        let credentials = self.read_credential().await?;
        let redirect_uri = self.auth.oauth_callback_url(OAuthProvider::Gitlab).await?;
        create_authorization_url(&credentials.client_id, &redirect_uri)
    }
}

fn create_authorization_url(client_id: &str, redirect_uri: &str) -> Result<String> {
    let mut url = reqwest::Url::parse("https://gitlab.com/oauth/authorize")?;
    let params = vec![
        ("client_id", client_id),
        ("response_type", "code"),
        ("scope", "api"),
        ("redirect_uri", redirect_uri),
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
        let url =
            create_authorization_url("client_id", "http://localhost:8080/oauth/callback/gitlab")
                .unwrap();
        assert_eq!(url, "https://gitlab.com/oauth/authorize?client_id=client_id&response_type=code&scope=api&redirect_uri=http%3A%2F%2Flocalhost%3A8080%2Foauth%2Fcallback%2Fgitlab");
    }
}
