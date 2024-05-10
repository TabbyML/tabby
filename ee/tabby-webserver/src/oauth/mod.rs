mod github;
mod google;

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use github::GithubClient;
use google::GoogleClient;
use tabby_schema::auth::{AuthenticationService, OAuthProvider};

#[async_trait]
pub trait OAuthClient: Send + Sync {
    async fn fetch_user_email(&self, code: String) -> Result<String>;
    async fn get_authorization_url(&self) -> Result<String>;
}

pub fn new_oauth_client(
    provider: OAuthProvider,
    auth: Arc<dyn AuthenticationService>,
) -> Arc<dyn OAuthClient> {
    match provider {
        OAuthProvider::Google => Arc::new(GoogleClient::new(auth)),
        OAuthProvider::Github => Arc::new(GithubClient::new(auth)),
    }
}
