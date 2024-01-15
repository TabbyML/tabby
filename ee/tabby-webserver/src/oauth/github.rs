use anyhow::Result;
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

#[derive(Default)]
pub struct GithubClient {
    client: reqwest::Client,
}

impl GithubClient {
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
            .await?
            .json::<Vec<GithubUserEmail>>()
            .await?;

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
        let params = [
            ("client_id", credential.client_id.as_str()),
            ("client_secret", credential.client_secret.as_str()),
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
