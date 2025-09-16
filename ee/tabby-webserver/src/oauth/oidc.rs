use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use cached::{proc_macro::cached, TimedCache};
use openidconnect::{
    core::{CoreAuthenticationFlow, CoreClient, CoreProviderMetadata, CoreUserInfoClaims},
    AccessTokenHash, AuthorizationCode, ClientId, ClientSecret, CsrfToken, IssuerUrl, Nonce,
    OAuth2TokenResponse, PkceCodeChallenge, PkceCodeVerifier, RedirectUrl, TokenResponse,
};
use serde::Deserialize;
use tabby_schema::auth::{AuthenticationService, OAuthCredential, OAuthProvider};
use tracing::error;

use super::OAuthClient;
use crate::bail;

pub struct OidcClient {
    auth: Arc<dyn AuthenticationService>,
    user_info: Mutex<Option<CoreUserInfoClaims>>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OAuthRequest {
    pub nonce: String,
    pub pkce_verifier: String,
}

static AUTH_REQS: LazyLock<Mutex<HashMap<String, OAuthRequest>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

impl OidcClient {
    pub fn new(auth: Arc<dyn AuthenticationService>) -> Self {
        Self {
            auth,
            user_info: Mutex::new(None),
        }
    }

    async fn read_credential(&self) -> Result<OAuthCredential> {
        match self.auth.read_oauth_credential(OAuthProvider::Oidc).await? {
            Some(credential) => Ok(credential),
            None => bail!("No OAuth credential found"),
        }
    }

    async fn retrieve_provider_metadata(&self, config_url: String) -> Option<CoreProviderMetadata> {
        retrieve_provider_metadata(config_url).await
    }
}

#[async_trait]
impl OAuthClient for OidcClient {
    async fn exchange_code_for_token(&self, code: String, state: Option<String>) -> Result<String> {
        let auth_req = {
            let mut auth_reqs = AUTH_REQS.lock().unwrap();
            match state {
                Some(state) => match auth_reqs.remove(&state) {
                    Some(auth_req_raw) => auth_req_raw,
                    None => bail!("Invalid authentication request."),
                },
                None => bail!("Invalid authentication state."),
            }
        };

        let credential = self.read_credential().await?;
        let config_url = match credential.config_url {
            Some(config_url) => config_url,
            None => bail!("No config url found."),
        };
        let provider_metadata = self.retrieve_provider_metadata(config_url).await.unwrap();
        let redirect_uri =
            RedirectUrl::new(self.auth.oauth_callback_url(OAuthProvider::Oidc).await?)?;
        let oidc_client = CoreClient::from_provider_metadata(
            provider_metadata,
            ClientId::new(credential.client_id),
            Some(ClientSecret::new(credential.client_secret)),
        )
        .set_redirect_uri(redirect_uri);

        let client = reqwest::Client::new();
        let pkce_verifier = PkceCodeVerifier::new(auth_req.pkce_verifier.clone());
        let token_response = oidc_client
            .exchange_code(AuthorizationCode::new(code))?
            .set_pkce_verifier(pkce_verifier)
            .request_async(&client)
            .await?;

        let id_token = token_response
            .id_token()
            .ok_or_else(|| anyhow!("Invalid authentication token"))?;

        let id_token_verifier = oidc_client.id_token_verifier();
        let nonce = Nonce::new(auth_req.nonce.clone());
        let claims = id_token.claims(&id_token_verifier, &nonce)?;

        if let Some(expected_access_token_hash) = claims.access_token_hash() {
            let actual_access_token_hash = AccessTokenHash::from_token(
                token_response.access_token(),
                id_token.signing_alg()?,
                id_token.signing_key(&id_token_verifier)?,
            )?;
            if actual_access_token_hash != *expected_access_token_hash {
                bail!("Invalid access token");
            }
        }

        let access_token = token_response.access_token().secret().to_string();

        // Get User info
        let user_info_response = oidc_client
            .user_info(token_response.access_token().to_owned(), None)?
            .request_async(&client)
            .await;

        let mut user_info = self.user_info.lock().unwrap();
        *user_info = user_info_response.ok();

        Ok(access_token)
    }

    async fn fetch_user_email(&self, _access_token: &str) -> Result<String> {
        let user_info = self.user_info.lock().unwrap();
        match &*user_info {
            Some(user_info) => {
                let end_user_email = user_info.email().unwrap().to_owned();
                let email = end_user_email.to_string();
                Ok(email)
            }
            None => bail!("User email not available"),
        }
    }

    async fn fetch_user_full_name(&self, _access_token: &str) -> Result<String> {
        let user_info = self.user_info.lock().unwrap();
        match &*user_info {
            Some(user_info) => {
                let end_user_full_name = user_info.name().unwrap().to_owned();
                let full_name = end_user_full_name.get(None).unwrap().to_string();
                Ok(full_name)
            }
            None => bail!("User full name not available"),
        }
    }

    async fn get_authorization_url(&self) -> Result<String> {
        let credential = self.read_credential().await?;
        let config_url = match credential.config_url {
            Some(config_url) => config_url,
            None => bail!("No config url found."),
        };
        let provider_metadata = match self.retrieve_provider_metadata(config_url).await {
            Some(provider_metadata) => provider_metadata,
            None => bail!("Error retrieving provider metadata"),
        };

        let redirect_uri =
            RedirectUrl::new(self.auth.oauth_callback_url(OAuthProvider::Oidc).await?)?;
        let scopes_supported = match credential.config_scopes {
            Some(config_scopes) => config_scopes
                .split_whitespace()
                .map(|s| s.to_string())
                .collect(),
            None => vec!["openid".into(), "profile".into(), "email".into()],
        };

        let oidc_client = CoreClient::from_provider_metadata(
            provider_metadata,
            ClientId::new(credential.client_id),
            Some(ClientSecret::new(credential.client_secret)),
        )
        .set_redirect_uri(redirect_uri);

        let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();

        let mut authorization_request = oidc_client
            .authorize_url(
                CoreAuthenticationFlow::AuthorizationCode,
                CsrfToken::new_random,
                Nonce::new_random,
            )
            .set_pkce_challenge(pkce_challenge);

        {
            authorization_request = authorization_request.add_scopes(
                scopes_supported
                    .iter()
                    .map(|s| openidconnect::Scope::new(s.to_string())),
            )
        }

        let (auth_uri, csrf_token, nonce) = authorization_request.url();
        let auth_req = OAuthRequest {
            nonce: nonce.secret().clone(),
            pkce_verifier: pkce_verifier.into_secret(),
        };

        let mut auth_reqs = AUTH_REQS.lock().unwrap();
        auth_reqs.insert(csrf_token.into_secret(), auth_req);

        Ok(auth_uri.to_string())
    }
}

#[cached(
    type = "TimedCache<String, Option<CoreProviderMetadata>>",
    create = "{ TimedCache::with_lifespan(3600 * 12) }"
)]
async fn retrieve_provider_metadata(config_url: String) -> Option<CoreProviderMetadata> {
    let client = reqwest::Client::new();
    let provider_metadata =
        CoreProviderMetadata::discover_async(IssuerUrl::new(config_url).ok().unwrap(), &client)
            .await;

    match provider_metadata {
        Ok(provider_metadata) => Some(provider_metadata),
        Err(e) => {
            error!("Failed to retrieve provider metadata: {}", e);
            None
        }
    }
}
