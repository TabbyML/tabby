use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::Deserialize;
use tabby_schema::auth::{AuthenticationService, OAuthCredential, OAuthProvider};

use super::OAuthClient;
use crate::bail;

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

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GithubUser {
    name: String,
}

pub struct GithubClient {
    client: reqwest::Client,
    auth: Arc<dyn AuthenticationService>,
}

impl GithubClient {
    pub fn new(auth: Arc<dyn AuthenticationService>) -> Self {
        Self {
            client: reqwest::Client::new(),
            auth,
        }
    }

    async fn read_credential(&self) -> Result<OAuthCredential> {
        match self
            .auth
            .read_oauth_credential(OAuthProvider::Github)
            .await?
        {
            Some(credential) => Ok(credential),
            None => bail!("No Github OAuth credential found"),
        }
    }

    async fn exchange_access_token(
        &self,
        code: String,
        credential: OAuthCredential,
    ) -> Result<GithubOAuthResponse> {
        let client_secret = credential.client_secret;

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
}

#[async_trait]
impl OAuthClient for GithubClient {
    async fn exchange_code_for_token(&self, code: String) -> Result<String> {
        let credentials = self.read_credential().await?;
        let token_resp = self.exchange_access_token(code, credentials).await?;
        if !token_resp.error.is_empty() {
            bail!(
                "Failed to exchange access token: {}",
                token_resp.error_description
            );
        }

        Ok(token_resp.access_token)
    }

    async fn fetch_user_email(&self, access_token: &str) -> Result<String> {
        let resp = self
            .client
            .get("https://api.github.com/user/emails")
            .header(reqwest::header::USER_AGENT, "Tabby")
            .header(reqwest::header::ACCEPT, "application/vnd.github+json")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {access_token}"),
            )
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await?;

        let emails = resp.json::<Vec<GithubUserEmail>>().await?;
        select_primary_email(emails)
    }

    async fn fetch_user_full_name(&self, access_token: &str) -> Result<String> {
        let resp = self
            .client
            .get("https://api.github.com/user")
            .header(reqwest::header::USER_AGENT, "Tabby")
            .header(reqwest::header::ACCEPT, "application/vnd.github+json")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {access_token}"),
            )
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await?;

        let user: serde_json::Value = resp.json().await?;
        let name = user["name"].as_str().unwrap_or_default();
        Ok(name.to_string())
    }

    async fn get_authorization_url(&self) -> Result<String> {
        let credentials = self.read_credential().await?;
        create_authorization_url(&credentials.client_id)
    }
}

fn create_authorization_url(client_id: &str) -> Result<String> {
    let mut url = reqwest::Url::parse("https://github.com/login/oauth/authorize")?;
    let params = vec![
        ("client_id", client_id),
        ("response_type", "code"),
        ("scope", "read:user user:email"),
    ];
    for (k, v) in params {
        url.query_pairs_mut().append_pair(k, v);
    }
    Ok(url.to_string())
}

fn select_primary_email(emails: Vec<GithubUserEmail>) -> Result<String> {
    emails
        .into_iter()
        .find(|item| item.primary)
        .map(|item| item.email)
        .ok_or(anyhow!("No primary email address found"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_authorization_url() {
        let url = create_authorization_url("client_id").unwrap();
        assert_eq!(url, "https://github.com/login/oauth/authorize?client_id=client_id&response_type=code&scope=read%3Auser+user%3Aemail");
    }

    #[test]
    fn test_select_primary_email() {
        let emails = vec![
            GithubUserEmail {
                email: "a@example.com".into(),
                primary: false,
                verified: true,
                visibility: None,
            },
            GithubUserEmail {
                email: "b@example.com".into(),
                primary: true,
                verified: true,
                visibility: None,
            },
            GithubUserEmail {
                email: "c@example.com".into(),
                primary: false,
                verified: true,
                visibility: None,
            },
        ];

        assert_eq!(select_primary_email(emails).unwrap(), "b@example.com");

        let emails = vec![GithubUserEmail {
            email: "a@example.com".into(),
            primary: false,
            verified: true,
            visibility: None,
        }];

        assert!(select_primary_email(emails).is_err());
    }
}
