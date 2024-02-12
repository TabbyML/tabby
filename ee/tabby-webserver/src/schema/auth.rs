use std::fmt::Debug;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use jsonwebtoken as jwt;
use juniper::{
    FieldError, GraphQLEnum, GraphQLInputObject, GraphQLObject, IntoFieldError, ScalarValue, ID,
};
use juniper_axum::relay;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use tabby_common::terminal::{HeaderFormat, InfoMessage};
use thiserror::Error;
use tracing::{error, warn};
use uuid::Uuid;
use validator::{Validate, ValidationErrors};

use super::from_validation_errors;
use crate::schema::Context;

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
pub enum TokenAuthError {
    #[error("Invalid input parameters")]
    InvalidInput(#[from] ValidationErrors),

    #[error("User not found")]
    UserNotFound,

    #[error("Password is not valid")]
    InvalidPassword,

    #[error("User is disabled")]
    UserDisabled,

    #[error(transparent)]
    Other(#[from] anyhow::Error),

    #[error("Unknown error")]
    Unknown,
}

#[derive(Default, Serialize)]
pub struct OAuthResponse {
    pub access_token: String,
    pub refresh_token: String,
}

#[derive(Error, Debug)]
pub enum OAuthError {
    #[error("The code passed is incorrect or expired")]
    InvalidVerificationCode,

    #[error("The credential is not active")]
    CredentialNotActive,

    #[error("The user is not invited to access the system")]
    UserNotInvited,

    #[error("User is disabled")]
    UserDisabled,

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

    #[error("User is disabled")]
    UserDisabled,

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
    pub claims: JWTPayload,
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
    pub active: bool,
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
    pub client_secret: String,
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
    async fn delete_expired_token(&self) -> Result<()>;
    async fn verify_access_token(&self, access_token: &str) -> Result<VerifyTokenResponse>;
    async fn is_admin_initialized(&self) -> Result<bool>;
    async fn get_user_by_email(&self, email: &str) -> Result<User>;

    async fn create_invitation(&self, email: String) -> Result<Invitation>;
    async fn delete_invitation(&self, id: &ID) -> Result<ID>;

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
