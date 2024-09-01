mod github;
mod gitlab;
mod google;

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use github::GithubClient;
use google::GoogleClient;
use tabby_schema::auth::{AuthenticationService, OAuthProvider};

use self::gitlab::GitlabClient;

#[async_trait]
pub trait OAuthClient: Send + Sync {
    async fn exchange_code_for_token(&self, code: String) -> Result<String>;
    async fn fetch_user_email(&self, access_token: &str) -> Result<String>;
    async fn fetch_user_full_name(&self, access_token: &str) -> Result<String>;
    async fn get_authorization_url(&self) -> Result<String>;
}

pub fn new_oauth_client(
    provider: OAuthProvider,
    auth: Arc<dyn AuthenticationService>,
) -> Arc<dyn OAuthClient> {
    match provider {
        OAuthProvider::Gitlab => Arc::new(GitlabClient::new(auth)),
        OAuthProvider::Google => Arc::new(GoogleClient::new(auth)),
        OAuthProvider::Github => Arc::new(GithubClient::new(auth)),
    }
}

#[cfg(test)]
pub mod test_client {
    use super::*;

    pub struct TestOAuthClient {
        pub access_token_response: fn() -> Result<String>,
        pub user_email: String,
        pub user_name: String,
    }

    #[async_trait]
    impl OAuthClient for TestOAuthClient {
        async fn exchange_code_for_token(&self, _code: String) -> Result<String> {
            (self.access_token_response)()
        }

        async fn fetch_user_email(&self, _access_token: &str) -> Result<String> {
            Ok(self.user_email.clone())
        }

        async fn fetch_user_full_name(&self, _access_token: &str) -> Result<String> {
            Ok(self.user_name.clone())
        }

        async fn get_authorization_url(&self) -> Result<String> {
            Ok("https://example.com".into())
        }
    }
}
