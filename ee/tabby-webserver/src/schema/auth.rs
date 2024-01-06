use std::{fmt::Debug, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use jsonwebtoken as jwt;
use juniper::{FieldError, GraphQLObject, IntoFieldError, ScalarValue, ID};
use juniper_axum::relay;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{error, warn};
use uuid::Uuid;
use validator::ValidationErrors;

use super::from_validation_errors;
use crate::{oauth::github::GithubClient, schema::Context};

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
        eprintln!("
    \x1b[93;1mJWT secret is not set\x1b[0m

    Tabby server will generate a one-time (non-persisted) JWT secret for the current process.
    Please set the \x1b[94mTABBY_WEBSERVER_JWT_TOKEN_SECRET\x1b[0m environment variable for production usage.
"
        );
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

#[derive(Error, Debug)]
pub enum RegisterError {
    #[error("Invalid input parameters")]
    InvalidInput(#[from] ValidationErrors),

    #[error("Invitation code is not valid")]
    InvalidInvitationCode,

    #[error("Email is already registered")]
    DuplicateEmail,

    #[error(transparent)]
    Other(#[from] anyhow::Error),

    #[error("Unknown error")]
    Unknown,
}

impl<S: ScalarValue> IntoFieldError<S> for RegisterError {
    fn into_field_error(self) -> FieldError<S> {
        match self {
            Self::InvalidInput(errors) => from_validation_errors(errors),
            _ => self.into(),
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

#[derive(Error, Debug)]
pub enum CoreError {
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Error, Debug)]
pub enum TokenAuthError {
    #[error("Invalid input parameters")]
    InvalidInput(#[from] ValidationErrors),

    #[error("User not found")]
    UserNotFound,

    #[error("Password is not valid")]
    InvalidPassword,

    #[error(transparent)]
    Other(#[from] anyhow::Error),

    #[error("Unknown error")]
    Unknown,
}

#[derive(Default, Serialize)]
pub struct GithubAuthResponse {
    pub access_token: String,
    pub refresh_token: String,
}

#[derive(Error, Debug)]
pub enum GithubAuthError {
    #[error("The code passed is incorrect or expired")]
    InvalidVerificationCode,

    #[error("The Github credential is not active")]
    CredentialNotActive,

    #[error(transparent)]
    Other(#[from] anyhow::Error),

    #[error("Unknown error")]
    Unknown,
}

impl<S: ScalarValue> IntoFieldError<S> for TokenAuthError {
    fn into_field_error(self) -> FieldError<S> {
        match self {
            Self::InvalidInput(errors) => from_validation_errors(errors),
            _ => self.into(),
        }
    }
}

#[derive(Error, Debug)]
pub enum RefreshTokenError {
    #[error("Invalid refresh token")]
    InvalidRefreshToken,

    #[error("Expired refresh token")]
    ExpiredRefreshToken,

    #[error("User not found")]
    UserNotFound,

    #[error(transparent)]
    Other(#[from] anyhow::Error),

    #[error("Unknown error")]
    Unknown,
}

impl<S: ScalarValue> IntoFieldError<S> for RefreshTokenError {
    fn into_field_error(self) -> FieldError<S> {
        self.into()
    }
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

#[derive(Debug, GraphQLObject)]
pub struct VerifyTokenResponse {
    claims: JWTPayload,
}

impl VerifyTokenResponse {
    pub fn new(claims: JWTPayload) -> Self {
        Self { claims }
    }
}

#[derive(Debug, Default, Serialize, Deserialize, GraphQLObject)]
pub struct JWTPayload {
    /// Expiration time (as UTC timestamp)
    exp: f64,

    /// Issued at (as UTC timestamp)
    iat: f64,

    /// User email address
    pub sub: String,

    /// Whether the user is admin.
    pub is_admin: bool,
}

impl JWTPayload {
    pub fn new(email: String, is_admin: bool) -> Self {
        let now = jwt::get_current_timestamp();
        Self {
            iat: now as f64,
            exp: (now + *JWT_DEFAULT_EXP) as f64,
            sub: email,
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
    pub auth_token: String,
    pub created_at: DateTime<Utc>,
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

#[deprecated]
#[derive(Debug, Default, Serialize, Deserialize, GraphQLObject)]
pub struct Invitation {
    pub id: i32,
    pub email: String,
    pub code: String,

    pub created_at: String,
}

impl From<InvitationNext> for Invitation {
    fn from(value: InvitationNext) -> Self {
        Self {
            id: value.id.parse::<i32>().unwrap(),
            email: value.email,
            code: value.code,
            created_at: value.created_at,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, GraphQLObject)]
#[graphql(context = Context)]
pub struct InvitationNext {
    pub id: juniper::ID,
    pub email: String,
    pub code: String,

    pub created_at: String,
}

impl relay::NodeType for InvitationNext {
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

#[async_trait]
pub trait AuthenticationService: Send + Sync {
    async fn register(
        &self,
        email: String,
        password1: String,
        password2: String,
        invitation_code: Option<String>,
    ) -> std::result::Result<RegisterResponse, RegisterError>;

    async fn token_auth(
        &self,
        email: String,
        password: String,
    ) -> std::result::Result<TokenAuthResponse, TokenAuthError>;

    async fn refresh_token(
        &self,
        refresh_token: String,
    ) -> std::result::Result<RefreshTokenResponse, RefreshTokenError>;
    async fn verify_access_token(&self, access_token: &str) -> Result<VerifyTokenResponse>;
    async fn is_admin_initialized(&self) -> Result<bool>;
    async fn get_user_by_email(&self, email: &str) -> Result<User>;

    async fn create_invitation(&self, email: String) -> Result<ID>;
    async fn delete_invitation(&self, id: ID) -> Result<ID>;

    async fn reset_user_auth_token(&self, email: &str) -> Result<()>;

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
    ) -> Result<Vec<InvitationNext>>;

    async fn github_auth(
        &self,
        code: String,
        client: Arc<GithubClient>,
    ) -> std::result::Result<GithubAuthResponse, GithubAuthError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_generate_jwt() {
        let claims = JWTPayload::new("test".to_string(), false);
        let token = generate_jwt(claims).unwrap();

        assert!(!token.is_empty())
    }

    #[test]
    fn test_validate_jwt() {
        let claims = JWTPayload::new("test".to_string(), false);
        let token = generate_jwt(claims).unwrap();
        let claims = validate_jwt(&token).unwrap();
        assert_eq!(claims.sub, "test");
        assert!(!claims.is_admin);
    }

    #[test]
    fn test_generate_refresh_token() {
        let token = generate_refresh_token();
        assert_eq!(token.len(), 32);
    }
}
