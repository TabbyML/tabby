use std::{borrow::Cow, fmt::Debug};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLInputObject, GraphQLObject, ID};
use serde::{Deserialize, Serialize};
use strum::EnumIter;
use thiserror::Error;
use tokio::task::JoinHandle;
use tracing::error;
use validator::Validate;

use super::interface::UserValue;
use crate::{
    juniper::relay,
    policy::AccessPolicy,
    schema::{Context, Result},
};

#[derive(Debug, GraphQLObject)]
pub struct RegisterResponse {
    access_token: String,
    pub refresh_token: String,
}

impl RegisterResponse {
    pub fn new(access_token: String, refresh_token: String) -> Self {
        Self {
            access_token,
            refresh_token,
        }
    }
}

#[derive(Debug, GraphQLObject)]
pub struct TokenAuthResponse {
    access_token: String,
    pub refresh_token: String,
}

impl TokenAuthResponse {
    pub fn new(access_token: String, refresh_token: String) -> Self {
        Self {
            access_token,
            refresh_token,
        }
    }
}

/// Input parameters for token_auth mutation
/// See `RegisterInput` for `validate` attribute usage
#[derive(Validate)]
pub struct TokenAuthInput {
    #[validate(email(code = "email", message = "Email is invalid"))]
    #[validate(length(
        max = 128,
        code = "email",
        message = "Email must be at most 128 characters"
    ))]
    pub email: String,
    #[validate(length(
        min = 8,
        max = 20,
        code = "password",
        message = "Password must be between 8 and 20 characters"
    ))]
    pub password: String,
}

/// Input parameters for token_auth_ldap mutation
#[derive(Validate)]
pub struct TokenAuthLdapInput<'a> {
    #[validate(length(min = 1, code = "user_id", message = "User ID should not be empty"))]
    pub user_id: &'a str,
    #[validate(length(min = 1, code = "password", message = "Password should not be empty"))]
    pub password: &'a str,
}

/// Input parameters for register mutation
/// `validate` attribute is used to validate the input parameters
///   - `code` argument specifies which parameter causes the failure
///   - `message` argument provides client friendly error message
///
#[derive(Validate)]
pub struct RegisterInput {
    #[validate(email(code = "email", message = "Email is invalid"))]
    #[validate(length(
        max = 128,
        code = "email",
        message = "Email must be at most 128 characters"
    ))]
    pub email: String,
    #[validate(length(
        min = 8,
        max = 20,
        code = "password1",
        message = "Password must be between 8 and 20 characters"
    ))]
    #[validate(custom(function = "validate_password"))]
    pub password1: String,
    #[validate(must_match(
        code = "password2",
        message = "Passwords do not match",
        other = "password1"
    ))]
    #[validate(length(
        max = 20,
        code = "password2",
        message = "Password must be at most 20 characters"
    ))]
    pub password2: String,
}

#[derive(Default, Serialize)]
pub struct OAuthResponse {
    pub access_token: String,
    pub refresh_token: String,
}

#[derive(Error, Debug)]
pub enum OAuthError {
    #[error("User is not invited, please contact admin for help")]
    UserNotInvited,

    #[error("User is disabled, please contact admin for help")]
    UserDisabled,

    #[error("Seat limit on license would be exceeded")]
    InsufficientSeats,

    #[error(transparent)]
    Other(#[from] anyhow::Error),

    #[error("Unknown error")]
    Unknown,
}

#[derive(Debug, GraphQLObject)]
pub struct RefreshTokenResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub refresh_expires_at: DateTime<Utc>,
}

impl RefreshTokenResponse {
    pub fn new(
        access_token: String,
        refresh_token: String,
        refresh_expires_at: DateTime<Utc>,
    ) -> Self {
        Self {
            access_token,
            refresh_token,
            refresh_expires_at,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JWTPayload {
    /// Expiration time (as UTC timestamp)
    exp: i64,

    /// Issued at (as UTC timestamp)
    iat: i64,

    /// User id string
    pub sub: ID,

    /// Whether the token is generated from auth token based authentication
    #[serde(skip)]
    pub is_generated_from_auth_token: bool,
}

impl JWTPayload {
    pub fn new(id: ID, iat: i64, exp: i64, is_generated_from_auth_token: bool) -> Self {
        Self {
            sub: id,
            iat,
            exp,
            is_generated_from_auth_token,
        }
    }
}

#[derive(Debug, GraphQLObject, Clone)]
#[graphql(context = Context, impl = [UserValue])]
pub struct UserSecured {
    // === implements User ===
    pub id: juniper::ID,
    pub email: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub is_admin: bool,
    pub is_owner: bool,
    pub active: bool,
    // === end User ===
    pub auth_token: String,
    pub is_password_set: bool,

    // is_sso_user is used to indicate if the user is created by SSO
    // and should not be able to change Name and Password
    // e.g. LDAP, OAuth users
    pub is_sso_user: bool,

    #[graphql(skip)]
    pub policy: AccessPolicy,
}

impl relay::NodeType for UserSecured {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "UserSecuredConnection"
    }

    fn edge_type_name() -> &'static str {
        "UserSecuredEdge"
    }
}

#[derive(Validate, GraphQLInputObject)]
pub struct RequestInvitationInput {
    #[validate(email(code = "email", message = "Invalid email address"))]
    pub email: String,
}

#[derive(Validate, GraphQLInputObject)]
pub struct RequestPasswordResetEmailInput {
    #[validate(email(code = "email", message = "Invalid email address"))]
    pub email: String,
}

#[derive(Validate, GraphQLInputObject)]
pub struct PasswordResetInput {
    pub code: String,
    #[validate(length(
        min = 8,
        max = 20,
        code = "password1",
        message = "Password must be between 8 and 20 characters"
    ))]
    #[validate(custom(function = "validate_password"))]
    pub password1: String,
    #[validate(length(
        min = 8,
        max = 20,
        code = "password2",
        message = "Password must be between 8 and 20 characters"
    ))]
    #[validate(must_match(
        code = "password2",
        message = "Passwords do not match",
        other = "password1"
    ))]
    pub password2: String,
}

#[derive(Validate, GraphQLInputObject)]
pub struct PasswordChangeInput {
    pub old_password: Option<String>,

    #[validate(length(
        min = 8,
        max = 20,
        code = "newPassword1",
        message = "Password must be between 8 and 20 characters"
    ))]
    #[validate(custom(function = "validate_new_password"))]
    pub new_password1: String,
    #[validate(length(
        min = 8,
        max = 20,
        code = "newPassword2",
        message = "Password must be between 8 and 20 characters"
    ))]
    #[validate(must_match(
        code = "newPassword2",
        message = "Passwords do not match",
        other = "new_password1"
    ))]
    pub new_password2: String,
}

#[derive(Validate)]
pub struct UpdateUserNameInput {
    #[validate(length(
        min = 2,
        max = 20,
        code = "name",
        message = "Name must be between 2 and 20 characters"
    ))]
    #[validate(regex(
        code = "name",
        path = "*crate::schema::constants::USERNAME_REGEX",
        message = "Invalid name, name may contain numbers or special characters which are not supported"
    ))]
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize, GraphQLObject)]
#[graphql(context = Context)]
pub struct Invitation {
    pub id: juniper::ID,
    pub email: String,
    pub code: String,

    pub created_at: DateTime<Utc>,
}

impl relay::NodeType for Invitation {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "InvitationConnection"
    }

    fn edge_type_name() -> &'static str {
        "InvitationEdge"
    }
}

#[derive(GraphQLEnum, Clone, Serialize, Deserialize, PartialEq, Debug, EnumIter)]
#[serde(rename_all = "lowercase")]
pub enum OAuthProvider {
    Github,
    Google,
    Gitlab,
}

#[derive(GraphQLEnum, Clone, Serialize, Deserialize, PartialEq, Debug)]
pub enum AuthProviderKind {
    OAuthGithub,
    OAuthGoogle,
    OAuthGitlab,
    Ldap,
}

impl From<OAuthProvider> for AuthProvider {
    fn from(provider: OAuthProvider) -> Self {
        match provider {
            OAuthProvider::Github => AuthProvider {
                kind: AuthProviderKind::OAuthGithub,
            },
            OAuthProvider::Google => AuthProvider {
                kind: AuthProviderKind::OAuthGoogle,
            },
            OAuthProvider::Gitlab => AuthProvider {
                kind: AuthProviderKind::OAuthGitlab,
            },
        }
    }
}

#[derive(GraphQLObject)]
pub struct AuthProvider {
    pub kind: AuthProviderKind,
}

#[derive(GraphQLObject)]
pub struct OAuthCredential {
    pub provider: OAuthProvider,
    pub client_id: String,

    #[graphql(skip)]
    pub client_secret: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(GraphQLInputObject, Validate)]
pub struct UpdateOAuthCredentialInput {
    pub provider: OAuthProvider,

    #[validate(length(min = 1, code = "clientId", message = "Client ID cannot be empty"))]
    pub client_id: String,

    #[validate(length(
        min = 1,
        code = "clientSecret",
        message = "Client secret cannot be empty"
    ))]
    pub client_secret: Option<String>,
}

#[derive(GraphQLEnum, PartialEq, Debug)]
pub enum LdapEncryptionKind {
    None,
    StartTLS,
    LDAPS,
}

#[derive(GraphQLInputObject, Validate)]
pub struct UpdateLdapCredentialInput {
    #[validate(length(
        min = 1,
        code = "host",
        message = "host should not be empty and should be a valid hostname or IP address"
    ))]
    pub host: String,
    pub port: i32,

    #[validate(length(min = 1, code = "bindDn", message = "bindDn cannot be empty"))]
    pub bind_dn: String,
    pub bind_password: Option<String>,

    #[validate(length(min = 1, code = "baseDn", message = "baseDn cannot be empty"))]
    pub base_dn: String,
    #[validate(length(
        min = 1,
        code = "userFilter",
        message = "userFilter cannot be empty, and should be in the format of `(uid=%s)`"
    ))]
    pub user_filter: String,

    pub encryption: LdapEncryptionKind,
    pub skip_tls_verify: bool,

    #[validate(length(
        min = 1,
        code = "emailAttribute",
        message = "emailAttribute cannot be empty"
    ))]
    pub email_attribute: String,
    // if name_attribute is None, we will use username as name
    pub name_attribute: Option<String>,
}

#[derive(GraphQLObject)]
pub struct LdapCredential {
    pub host: String,
    pub port: i32,
    pub bind_dn: String,
    pub base_dn: String,
    pub user_filter: String,
    pub encryption: LdapEncryptionKind,
    pub skip_tls_verify: bool,
    pub email_attribute: String,
    pub name_attribute: Option<String>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[async_trait]
pub trait AuthenticationService: Send + Sync {
    async fn register(
        &self,
        email: String,
        password1: String,
        invitation_code: Option<String>,
        name: Option<String>,
    ) -> Result<RegisterResponse>;
    async fn allow_self_signup(&self) -> Result<bool>;

    async fn token_auth(&self, email: String, password: String) -> Result<TokenAuthResponse>;

    async fn token_auth_ldap(&self, email: &str, password: &str) -> Result<TokenAuthResponse>;

    async fn refresh_token(&self, refresh_token: String) -> Result<RefreshTokenResponse>;
    async fn verify_access_token(&self, access_token: &str) -> Result<JWTPayload>;
    async fn verify_auth_token(&self, token: &str) -> Result<ID>;
    async fn is_admin_initialized(&self) -> Result<bool>;
    async fn get_user_by_email(&self, email: &str) -> Result<UserSecured>;
    async fn get_user(&self, id: &ID) -> Result<UserSecured>;
    async fn logout_all_sessions(&self, id: &ID) -> Result<()>;

    async fn create_invitation(&self, email: String) -> Result<Invitation>;
    async fn request_invitation_email(&self, input: RequestInvitationInput) -> Result<Invitation>;
    async fn delete_invitation(&self, id: &ID) -> Result<ID>;

    async fn reset_user_auth_token(&self, id: &ID) -> Result<()>;
    async fn password_reset(&self, code: &str, password: &str) -> Result<()>;
    async fn generate_reset_password_url(&self, id: &ID) -> Result<String>;
    async fn request_password_reset_email(&self, email: String) -> Result<Option<JoinHandle<()>>>;
    async fn update_user_password(
        &self,
        id: &ID,
        old_password: Option<&str>,
        new_password: &str,
    ) -> Result<()>;

    async fn list_users(
        &self,
        ids: Option<Vec<ID>>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<UserSecured>>;

    async fn list_invitations(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Invitation>>;

    async fn oauth(
        &self,
        code: String,
        provider: OAuthProvider,
    ) -> std::result::Result<OAuthResponse, OAuthError>;

    async fn oauth_callback_url(&self, provider: OAuthProvider) -> Result<String>;

    async fn read_oauth_credential(
        &self,
        provider: OAuthProvider,
    ) -> Result<Option<OAuthCredential>>;

    async fn update_oauth_credential(&self, input: UpdateOAuthCredentialInput) -> Result<()>;
    async fn delete_oauth_credential(&self, provider: OAuthProvider) -> Result<()>;

    async fn read_ldap_credential(&self) -> Result<Option<LdapCredential>>;
    async fn test_ldap_connection(&self, input: UpdateLdapCredentialInput) -> Result<()>;
    async fn update_ldap_credential(&self, input: UpdateLdapCredentialInput) -> Result<()>;
    async fn delete_ldap_credential(&self) -> Result<()>;

    async fn update_user_active(&self, id: &ID, active: bool) -> Result<()>;
    async fn update_user_role(&self, id: &ID, is_admin: bool) -> Result<()>;
    async fn update_user_avatar(&self, id: &ID, avatar: Option<Box<[u8]>>) -> Result<()>;
    async fn get_user_avatar(&self, id: &ID) -> Result<Option<Box<[u8]>>>;
    async fn update_user_name(&self, id: &ID, name: String) -> Result<()>;
}

fn validate_password(value: &str) -> Result<(), validator::ValidationError> {
    validate_password_impl(value, "password1")
}

fn validate_new_password(value: &str) -> Result<(), validator::ValidationError> {
    validate_password_impl(value, "newPassword1")
}

fn validate_password_impl(
    value: &str,
    code: &'static str,
) -> Result<(), validator::ValidationError> {
    let make_validation_error = |message: &'static str| {
        let mut err = validator::ValidationError::new(code);
        err.message = Some(Cow::Borrowed(message));
        Err(err)
    };

    let contains_lowercase = value.chars().any(|x| x.is_ascii_lowercase());
    if !contains_lowercase {
        return make_validation_error("Password should contain at least one lowercase character");
    }

    let contains_uppercase = value.chars().any(|x| x.is_ascii_uppercase());
    if !contains_uppercase {
        return make_validation_error("Password should contain at least one uppercase character");
    }

    let contains_digit = value.chars().any(|x| x.is_ascii_digit());
    if !contains_digit {
        return make_validation_error("Password should contain at least one numeric character");
    }

    let contains_special_char = value.chars().any(|x| x.is_ascii_punctuation());
    if !contains_special_char {
        return make_validation_error(
            "Password should contain at least one special character, e.g @#$%^&{}",
        );
    }

    Ok(())
}
