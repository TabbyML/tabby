use anyhow::Result;
use hyper::{client::HttpConnector, Client, Method, Request};
use hyper_rustls::HttpsConnector;
use serde::Deserialize;
use tabby_db::GithubOAuthCredentialDAO;

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

pub struct GithubClient {
    client: Client<HttpsConnector<HttpConnector>>,
}

impl Default for GithubClient {
    fn default() -> Self {
        Self::new()
    }
}

impl GithubClient {
    pub fn new() -> Self {
        let https = hyper_rustls::HttpsConnectorBuilder::new()
            .with_native_roots()
            .expect("no native root CA certificates found")
            .https_only()
            .enable_all_versions()
            .build();
        let client: Client<_, hyper::Body> = Client::builder().build(https);
        Self { client }
    }

    pub async fn fetch_user_email(
        &self,
        code: String,
        credential: GithubOAuthCredentialDAO,
    ) -> Result<String> {
        let token_resp = self.exchange_access_token(code, credential).await?;
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
            .header(
                hyper::header::AUTHORIZATION,
                format!("Bearer {}", token_resp.access_token),
            )
            .header("X-GitHub-Api-Version", "2022-11-28")
            .body(hyper::Body::empty())?;
        let raw = self.client.request(req).await?;
        let body = hyper::body::to_bytes(raw.into_body()).await?;
        let resp = serde_json::from_slice::<Vec<GithubUserEmail>>(body.as_ref())?;

        if resp.is_empty() {
            return Err(anyhow::anyhow!("No email address found"));
        }
        for item in &resp {
            if item.primary {
                return Ok(item.email.clone());
            }
        }
        Ok(resp[0].email.clone())
    }

    async fn exchange_access_token(
        &self,
        code: String,
        credential: GithubOAuthCredentialDAO,
    ) -> Result<GithubOAuthResponse> {
        let req = Request::builder()
            .method(Method::POST)
            .uri("https://github.com/login/oauth/access_token")
            .header(hyper::header::ACCEPT, "application/json")
            .body(hyper::Body::from(format!(
                "client_id={}&client_secret={}&code={}",
                credential.client_id, credential.client_secret, code
            )))?;

        let raw = self.client.request(req).await?;
        let body = hyper::body::to_bytes(raw.into_body()).await?;
        let resp = serde_json::from_slice::<GithubOAuthResponse>(body.as_ref())?;

        Ok(resp)
    }
}
