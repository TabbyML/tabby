use std::{borrow::Cow, fmt::Debug};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use jsonwebtoken as jwt;
use juniper::{GraphQLEnum, GraphQLInputObject, GraphQLObject, ID};
use juniper_axum::relay;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use tabby_common::terminal::{HeaderFormat, InfoMessage};
use thiserror::Error;
use tokio::task::JoinHandle;
use tracing::{error, warn};
use uuid::Uuid;
use validator::Validate;

use crate::schema::{Context, Result};

lazy_static! {
    static ref JWT_TOKEN_SECRET: String  = jwt_token_secret();

    static ref JWT_ENCODING_KEY: jwt::EncodingKey = jwt::EncodingKey::from_secret(
        JWT_TOKEN_SECRET.as_bytes()
    );
    static ref JWT_DECODING_KEY: jwt::DecodingKey = jwt::DecodingKey::from_secret(
        JWT_TOKEN_SECRET.as_bytes()
    );
    static ref JWT_DEFAULT_EXP: u64 = 30 * 60; // 30 minutes
}

pub fn generate_jwt(claims: JWTPayload) -> jwt::errors::Result<String> {
    let header = jwt::Header::default();
    let token = jwt::encode(&header, &claims, &JWT_ENCODING_KEY)?;
    Ok(token)
}

pub fn validate_jwt(token: &str) -> jwt::errors::Result<JWTPayload> {
    let validation = jwt::Validation::default();
    let data = jwt::decode::<JWTPayload>(token, &JWT_DECODING_KEY, &validation)?;
    Ok(data.claims)
}

fn jwt_token_secret() -> String {
    let jwt_secret = std::env::var("TABBY_WEBSERVER_JWT_TOKEN_SECRET").unwrap_or_else(|_| {
        InfoMessage::new("JWT secret is not set", HeaderFormat::BoldYellow, &[
            "Tabby server will generate a one-time (non-persisted) JWT secret for the current process.",
            &format!("Please set the {} environment variable for production usage.", HeaderFormat::Blue.format("TABBY_WEBSERVER_JWT_TOKEN_SECRET")),
        ]).print();
        Uuid::new_v4().to_string()
    });

    if Uuid::parse_str(&jwt_secret).is_err() {
        warn!("JWT token secret needs to be in standard uuid format to ensure its security, you might generate one at https://www.uuidgenerator.net");
        std::process::exit(1)
    }

    jwt_secret
}

pub fn generate_refresh_token() -> String {
    Uuid::new_v4().to_string().replace('-', "")
}

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
        code = "password",
        message = "Password must be at least 8 characters"
    ))]
    #[validate(length(
        max = 20,
        code = "password",
        message = "Password must be at most 20 characters"
    ))]
    pub password: String,
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
        code = "password1",
        message = "Password must be at least 8 characters"
    ))]
    #[validate(length(
        max = 20,
        code = "password1",
        message = "Password must be at most 20 characters"
    ))]
    #[validate(custom = "validate_password")]
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
    #[error("The oauth code passed is incorrect or expired")]
    InvalidVerificationCode,

    #[error("OAuth is not enabled")]
    CredentialNotActive,

    #[error("User is not invited, please contact admin for help")]
    UserNotInvited,

    #[error("User is disabled, please contact admin for help")]
    UserDisabled,

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

// IDWrapper to used as a type guard for refactoring, can be removed in a follow up PR.
// FIXME(meng): refactor out IDWrapper.
#[derive(Serialize, Deserialize, Debug)]
pub struct IDWrapper(pub ID);

#[derive(Debug, Serialize, Deserialize)]
pub struct JWTPayload {
    /// Expiration time (as UTC timestamp)
    exp: i64,

    /// Issued at (as UTC timestamp)
    iat: i64,

    /// User id string
    pub sub: IDWrapper,

    /// Whether the user is admin.
    pub is_admin: bool,
}

impl JWTPayload {
    pub fn new(id: ID, is_admin: bool) -> Self {
        let now = jwt::get_current_timestamp();
        Self {
            iat: now as i64,
            exp: (now + *JWT_DEFAULT_EXP) as i64,
            sub: IDWrapper(id),
            is_admin,
        }
    }
}

#[derive(Debug, GraphQLObject)]
#[graphql(context = Context)]
pub struct User {
    pub id: juniper::ID,
    pub email: String,
    pub is_admin: bool,
    pub is_owner: bool,
    pub auth_token: String,
    pub created_at: DateTime<Utc>,
    pub active: bool,
    pub is_password_set: bool,
}

impl relay::NodeType for User {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "UserConnection"
    }

    fn edge_type_name() -> &'static str {
        "UserEdge"
    }
}

#[derive(Validate, GraphQLInputObject)]
pub struct RequestInvitationInput {
    #[validate(email(code = "email"))]
    pub email: String,
}

#[derive(Validate, GraphQLInputObject)]
pub struct RequestPasswordResetEmailInput {
    #[validate(email(code = "email"))]
    pub email: String,
}

#[derive(Validate, GraphQLInputObject)]
pub struct PasswordResetInput {
    pub code: String,
    #[validate(length(
        min = 8,
        code = "password1",
        message = "Password must be at least 8 characters"
    ))]
    #[validate(length(
        max = 20,
        code = "password1",
        message = "Password must be at most 20 characters"
    ))]
    pub password1: String,
    #[validate(length(
        min = 8,
        code = "password2",
        message = "Password must be at least 8 characters"
    ))]
    #[validate(length(
        max = 20,
        code = "password2",
        message = "Password must be at most 20 characters"
    ))]
    #[validate(must_match(
        code = "password2",
        message = "Passwords do not match",
        other = "password1"
    ))]
    pub password2: String,
}

#[derive(Validate, GraphQLInputObject)]
pub struct PasswordUpdateInput {
    pub old_password: Option<String>,

    #[validate(length(
        min = 8,
        code = "new_password1",
        message = "Password must be at least 8 characters"
    ))]
    #[validate(length(
        max = 20,
        code = "new_password1",
        message = "Password must be at most 20 characters"
    ))]
    pub new_password1: String,
    #[validate(length(
        min = 8,
        code = "new_password2",
        message = "Password must be at least 8 characters"
    ))]
    #[validate(length(
        max = 20,
        code = "new_password2",
        message = "Password must be at most 20 characters"
    ))]
    #[validate(must_match(
        code = "new_password2",
        message = "Passwords do not match",
        other = "new_password1"
    ))]
    pub new_password2: String,
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

#[derive(GraphQLEnum, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OAuthProvider {
    Github,
    Google,
}

#[derive(GraphQLObject)]
pub struct OAuthCredential {
    pub provider: OAuthProvider,
    pub client_id: String,

    pub client_secret: Option<String>,
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

#[async_trait]
pub trait AuthenticationService: Send + Sync {
    async fn register(
        &self,
        email: String,
        password1: String,
        invitation_code: Option<String>,
    ) -> Result<RegisterResponse>;
    async fn allow_self_signup(&self) -> Result<bool>;

    async fn token_auth(&self, email: String, password: String) -> Result<TokenAuthResponse>;

    async fn refresh_token(&self, refresh_token: String) -> Result<RefreshTokenResponse>;
    async fn delete_expired_token(&self) -> Result<()>;
    async fn delete_expired_password_resets(&self) -> Result<()>;
    async fn verify_access_token(&self, access_token: &str) -> Result<JWTPayload>;
    async fn is_admin_initialized(&self) -> Result<bool>;
    async fn get_user_by_email(&self, email: &str) -> Result<User>;
    async fn get_user(&self, id: &ID) -> Result<User>;

    async fn create_invitation(&self, email: String) -> Result<Invitation>;
    async fn request_invitation_email(&self, input: RequestInvitationInput) -> Result<Invitation>;
    async fn delete_invitation(&self, id: &ID) -> Result<ID>;

    async fn reset_user_auth_token(&self, id: &ID) -> Result<()>;
    async fn password_reset(&self, code: &str, password: &str) -> Result<()>;
    async fn request_password_reset_email(&self, email: String) -> Result<Option<JoinHandle<()>>>;
    async fn update_user_password(
        &self,
        id: &ID,
        old_password: Option<&str>,
        new_password: &str,
    ) -> Result<()>;

    async fn list_users(
        &self,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<User>>;

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
    async fn update_user_active(&self, id: &ID, active: bool) -> Result<()>;
    async fn update_user_role(&self, id: &ID, is_admin: bool) -> Result<()>;
}

fn validate_password(value: &str) -> Result<(), validator::ValidationError> {
    let make_validation_error = |message: &'static str| {
        let mut err = validator::ValidationError::new("password1");
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_generate_jwt() {
        let claims = JWTPayload::new(ID::from("test".to_owned()), false);
        let token = generate_jwt(claims).unwrap();

        assert!(!token.is_empty())
    }

    #[test]
    fn test_validate_jwt() {
        let claims = JWTPayload::new(ID::from("test".to_owned()), false);
        let token = generate_jwt(claims).unwrap();
        let claims = validate_jwt(&token).unwrap();
        assert_eq!(claims.sub.0.to_string(), "test");
        assert!(!claims.is_admin);
    }

    #[test]
    fn test_generate_refresh_token() {
        let token = generate_refresh_token();
        assert_eq!(token.len(), 32);
    }
}
