use async_trait::async_trait;
use chrono::{Duration, Utc};
use juniper::ID;
use tabby_schema::{
    auth::{
        AuthenticationService, Invitation, JWTPayload, LdapCredential, OAuthCredential, OAuthError,
        OAuthProvider, OAuthResponse, RefreshTokenResponse, RegisterResponse,
        RequestInvitationInput, TokenAuthResponse, UpdateLdapCredentialInput,
        UpdateOAuthCredentialInput, UserSecured,
    },
    Result,
};
use tokio::task::JoinHandle;

use crate::ldap::{LdapClient, LdapUser};

pub struct FakeLdapClient<'a> {
    pub state: &'a str,
}

#[async_trait]
impl LdapClient for FakeLdapClient<'_> {
    async fn validate(&mut self, user_id: &str, _password: &str) -> Result<LdapUser> {
        match self.state {
            "not_found" => Err(ldap3::LdapError::LdapResult {
                result: ldap3::LdapResult {
                    rc: 32,
                    matched: user_id.to_string(),
                    text: "User not found".to_string(),
                    refs: vec![],
                    ctrls: vec![],
                },
            }
            .into()),
            _ => Ok(LdapUser {
                email: "user@example.com".to_string(),
                name: "Test User".to_string(),
            }),
        }
    }
}

pub struct FakeAuthService {
    users: Vec<UserSecured>,
}

impl FakeAuthService {
    pub fn new(users: Vec<UserSecured>) -> Self {
        FakeAuthService { users }
    }
}

#[async_trait]
impl AuthenticationService for FakeAuthService {
    async fn register(
        &self,
        _email: String,
        _password: String,
        _invitation_code: Option<String>,
        _name: Option<String>,
    ) -> Result<RegisterResponse> {
        Ok(RegisterResponse::new(
            "access_token".to_string(),
            "refresh_token".to_string(),
        ))
    }

    async fn allow_self_signup(&self) -> Result<bool> {
        Ok(true)
    }

    async fn generate_reset_password_url(&self, id: &ID) -> Result<String> {
        Ok(format!("https://example.com/reset-password/{}", id))
    }

    async fn request_password_reset_email(&self, _email: String) -> Result<Option<JoinHandle<()>>> {
        Ok(None)
    }

    async fn password_reset(&self, _code: &str, _password: &str) -> Result<()> {
        Ok(())
    }

    async fn update_user_password(
        &self,
        _id: &ID,
        _old_password: Option<&str>,
        _new_password: &str,
    ) -> Result<()> {
        Ok(())
    }

    async fn update_user_avatar(&self, _id: &ID, _avatar: Option<Box<[u8]>>) -> Result<()> {
        Ok(())
    }

    async fn get_user_avatar(&self, _id: &ID) -> Result<Option<Box<[u8]>>> {
        Ok(None)
    }

    async fn update_user_name(&self, _id: &ID, _name: String) -> Result<()> {
        Ok(())
    }

    async fn token_auth(&self, _email: String, _password: String) -> Result<TokenAuthResponse> {
        Ok(TokenAuthResponse::new(
            "access_token".to_string(),
            "refresh_token".to_string(),
        ))
    }

    async fn token_auth_ldap(&self, _user_id: &str, _password: &str) -> Result<TokenAuthResponse> {
        Ok(TokenAuthResponse::new(
            "access_token".to_string(),
            "refresh_token".to_string(),
        ))
    }

    async fn refresh_token(&self, _token: String) -> Result<RefreshTokenResponse> {
        Ok(RefreshTokenResponse::new(
            "access_token".to_string(),
            "new_refresh_token".to_string(),
            Utc::now() + Duration::days(30),
        ))
    }

    async fn verify_access_token(&self, _access_token: &str) -> Result<JWTPayload> {
        Ok(JWTPayload::new(
            ID::new("user_id"),
            Utc::now().timestamp(),
            Utc::now().timestamp() + Duration::days(30).num_seconds(),
            false,
        ))
    }

    async fn verify_auth_token(&self, _token: &str) -> Result<ID> {
        Ok(ID::new("user_id"))
    }

    async fn is_admin_initialized(&self) -> Result<bool> {
        Ok(true)
    }

    async fn update_user_role(&self, _id: &ID, _is_admin: bool) -> Result<()> {
        Ok(())
    }

    async fn get_user_by_email(&self, email: &str) -> Result<UserSecured> {
        self.users
            .iter()
            .find(|user| user.email == email)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("User not found"))
            .map_err(Into::into)
    }

    async fn get_user(&self, id: &ID) -> Result<UserSecured> {
        self.users
            .iter()
            .find(|user| user.id == *id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("User not found"))
            .map_err(Into::into)
    }

    async fn create_invitation(&self, email: String) -> Result<Invitation> {
        let invitation = Invitation {
            id: ID::new("1"),
            email: email.clone(),
            code: "invitation_code".to_string(),
            created_at: Utc::now(),
        };
        Ok(invitation)
    }

    async fn request_invitation_email(&self, input: RequestInvitationInput) -> Result<Invitation> {
        self.create_invitation(input.email).await
    }

    async fn delete_invitation(&self, id: &ID) -> Result<ID> {
        Ok(id.clone())
    }

    async fn reset_user_auth_token(&self, _id: &ID) -> Result<()> {
        Ok(())
    }

    async fn logout_all_sessions(&self, _id: &ID) -> Result<()> {
        Ok(())
    }

    async fn list_users(
        &self,
        _ids: Option<Vec<ID>>,
        _after: Option<String>,
        _before: Option<String>,
        _first: Option<usize>,
        _last: Option<usize>,
    ) -> Result<Vec<UserSecured>> {
        Ok(self.users.clone())
    }

    async fn list_invitations(
        &self,
        _after: Option<String>,
        _before: Option<String>,
        _first: Option<usize>,
        _last: Option<usize>,
    ) -> Result<Vec<Invitation>> {
        Ok(vec![])
    }

    async fn read_ldap_credential(&self) -> Result<Option<LdapCredential>> {
        Ok(None)
    }

    async fn test_ldap_connection(&self, _credential: UpdateLdapCredentialInput) -> Result<()> {
        Ok(())
    }

    async fn update_ldap_credential(&self, _input: UpdateLdapCredentialInput) -> Result<()> {
        Ok(())
    }

    async fn delete_ldap_credential(&self) -> Result<()> {
        Ok(())
    }

    async fn oauth(
        &self,
        _code: String,
        _provider: OAuthProvider,
    ) -> std::result::Result<OAuthResponse, OAuthError> {
        Ok(OAuthResponse {
            access_token: "access_token".to_string(),
            refresh_token: "refresh_token".to_string(),
        })
    }

    async fn read_oauth_credential(
        &self,
        _provider: OAuthProvider,
    ) -> Result<Option<OAuthCredential>> {
        Ok(None)
    }

    async fn oauth_callback_url(&self, _provider: OAuthProvider) -> Result<String> {
        Ok("https://example.com/oauth/callback/".to_string())
    }

    async fn update_oauth_credential(&self, _input: UpdateOAuthCredentialInput) -> Result<()> {
        Ok(())
    }

    async fn delete_oauth_credential(&self, _provider: OAuthProvider) -> Result<()> {
        Ok(())
    }

    async fn update_user_active(&self, _id: &ID, _active: bool) -> Result<()> {
        Ok(())
    }
}
