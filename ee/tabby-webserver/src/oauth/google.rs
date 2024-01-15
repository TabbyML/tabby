use anyhow::Result;
use serde::Deserialize;
use tabby_db::GoogleOAuthCredentialDAO;

#[derive(Debug, Deserialize)]
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
        credential: GoogleOAuthCredentialDAO,
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
        credential: GoogleOAuthCredentialDAO,
    ) -> Result<GoogleOAuthResponse> {
        let params = [
            ("client_id", credential.client_id.as_str()),
            ("client_secret", credential.client_secret.as_str()),
            ("code", code.as_str()),
            ("grant_type", "authorization_code"),
            ("redirect_uri", credential.redirect_uri.as_str()),
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
