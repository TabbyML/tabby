use std::sync::{Arc, Mutex};

use anyhow::anyhow;
use anyhow::Result;
use async_trait::async_trait;
use openidconnect::{
    CsrfToken,
    ClientId,
    ClientSecret,
    DiscoveryError,
    IssuerUrl,
    Nonce,
    PkceCodeChallenge,
    RedirectUrl,
};
use openidconnect::core::{
    CoreAuthenticationFlow,
    CoreClient,
    CoreProviderMetadata,
};
use serde::Deserialize;
use tabby_schema::auth::{AuthenticationService, OAuthCredential, OAuthProvider};

use super::OAuthClient;
use crate::bail;

pub struct GeneralClient {
    client: reqwest::Client,
    auth: Arc<dyn AuthenticationService>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OAuthConfig {
    authorization_endpoint: String,
    token_endpoint: String,
    userinfo_endpoint: String,
    scopes_supported: Vec<String>,
}

impl GeneralClient {
    pub fn new(auth: Arc<dyn AuthenticationService>) -> Self {
        Self {
            client: reqwest::Client::new(),
            auth
        }
    }

    async fn read_credential(&self) -> Result<OAuthCredential> {
        match self
            .auth
            .read_oauth_credential(OAuthProvider::General)
            .await?
        {
            Some(credential) => Ok(credential),
            None => bail!("No OAuth credential found"),
        }
    }

    // TODO: Ensure that the HTTP client *does not* follow redirects.
    // TODO: Cache the HTTP response so we do not hit the endpoint every time we need the OIDC Discovery Endpoint
    async fn retrieve_provider_metadata(&self, config_url: Option<String>) ->Result<CoreProviderMetadata, anyhow::Error> {
        let config_url = config_url.unwrap_or_else(|| "".to_owned());

        // TODO: Add error handling for the HTTP client and the OIDC Discovery Endpoint (when the endpoint is not reachable)
        let client = reqwest::Client::new();
        let provider_metadata = CoreProviderMetadata::discover_async(
            IssuerUrl::new(config_url).ok().unwrap(),
            &client,
        ).await;

        match provider_metadata {
            Ok(provider_metadata) => Ok(provider_metadata),
            Err(e) => Err(anyhow!(e)),
        }
    }
}

#[async_trait]
impl OAuthClient for GeneralClient {
    async fn exchange_code_for_token(&self, code: String) -> Result<String> {
        Ok("".to_owned())
    }

    async fn fetch_user_email(&self, access_token: &str) -> Result<String> {
        Ok("".to_owned())
    }

    async fn fetch_user_full_name(&self, access_token: &str) -> Result<String> {
        Ok("".to_owned())
    }

    async fn get_authorization_url(&self) -> Result<String> {
        let credential = self.read_credential().await?;
        let config_url = credential.config_url;
        let oidc_config = match self.retrieve_provider_metadata(config_url).await
        {
            Ok(config) => config,
            Err(err) => return Err(err),
        };

        let redirect_uri = RedirectUrl::new(
            self.auth.oauth_callback_url(OAuthProvider::General).await?
        )?;

        let scopes_supported = oidc_config.scopes_supported().unwrap().clone();

        let oidc_client = CoreClient::from_provider_metadata(
            oidc_config,
            ClientId::new(credential.client_id),
            Some(ClientSecret::new(credential.client_secret)),
        ).set_redirect_uri(redirect_uri);

        let (pkce_chanllenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();

        let mut authorization_request = oidc_client
            .authorize_url(
                CoreAuthenticationFlow::AuthorizationCode,
                CsrfToken::new_random,
                Nonce::new_random,
            )
            .set_pkce_challenge(pkce_chanllenge);

        {
            authorization_request = authorization_request.add_scopes(
                scopes_supported
                    .iter()
                    .map(|s| openidconnect::Scope::new(s.to_string())),
            )
        }

        // TODO: Store the pkce_verifier in the session
        let (auth_uri, csrf_token, nonce) = authorization_request.url();
        Ok(auth_uri.to_string())
    }
}
