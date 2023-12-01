use std::fmt::Debug;

use anyhow::Result;
use async_trait::async_trait;
use jsonwebtoken as jwt;
use juniper::{FieldResult, GraphQLObject};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

lazy_static! {
    static ref JWT_ENCODING_KEY: jwt::EncodingKey = jwt::EncodingKey::from_secret(
        jwt_token_secret().as_bytes()
    );
    static ref JWT_DECODING_KEY: jwt::DecodingKey = jwt::DecodingKey::from_secret(
        jwt_token_secret().as_bytes()
    );
    static ref JWT_DEFAULT_EXP: u64 = 30 * 60; // 30 minutes
}

pub fn generate_jwt(claims: Claims) -> jwt::errors::Result<String> {
    let header = jwt::Header::default();
    let token = jwt::encode(&header, &claims, &JWT_ENCODING_KEY)?;
    Ok(token)
}

pub fn validate_jwt(token: &str) -> jwt::errors::Result<Claims> {
    let validation = jwt::Validation::default();
    let data = jwt::decode::<Claims>(token, &JWT_DECODING_KEY, &validation)?;
    Ok(data.claims)
}

fn jwt_token_secret() -> String {
    std::env::var("TABBY_WEBSERVER_JWT_TOKEN_SECRET").unwrap_or("default_secret".to_string())
}

#[derive(Debug, GraphQLObject)]
pub struct RegisterResponse {
    access_token: String,
    refresh_token: String,
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
    refresh_token: String,
}

impl TokenAuthResponse {
    pub fn new(access_token: String, refresh_token: String) -> Self {
        Self {
            access_token,
            refresh_token,
        }
    }
}

#[derive(Debug, Default, GraphQLObject)]
pub struct RefreshTokenResponse {
    access_token: String,
    refresh_token: String,
    refresh_expires_in: i32,
}

#[derive(Debug, GraphQLObject)]
pub struct VerifyTokenResponse {
    claims: Claims,
}

impl VerifyTokenResponse {
    pub fn new(claims: Claims) -> Self {
        Self { claims }
    }
}

#[derive(Debug, Default, PartialEq, Serialize, Deserialize, GraphQLObject)]
pub struct UserInfo {
    email: String,
    is_admin: bool,
}

impl UserInfo {
    pub fn new(email: String, is_admin: bool) -> Self {
        Self { email, is_admin }
    }

    pub fn is_admin(&self) -> bool {
        self.is_admin
    }

    pub fn email(&self) -> &str {
        &self.email
    }
}

#[derive(Debug, Default, Serialize, Deserialize, GraphQLObject)]
pub struct Claims {
    // Required. Expiration time (as UTC timestamp)
    exp: f64,
    // Optional. Issued at (as UTC timestamp)
    iat: f64,
    // Customized. user info
    user: UserInfo,
}

impl Claims {
    pub fn new(user: UserInfo) -> Self {
        let now = jwt::get_current_timestamp();
        Self {
            iat: now as f64,
            exp: (now + *JWT_DEFAULT_EXP) as f64,
            user,
        }
    }

    pub fn user_info(&self) -> &UserInfo {
        &self.user
    }
}

#[derive(Debug, Default, Serialize, Deserialize, GraphQLObject)]
pub struct Invitation {
    pub id: i32,
    pub email: String,
    pub code: String,

    pub created_at: String,
}

#[async_trait]
pub trait AuthenticationService: Send + Sync {
    async fn register(
        &self,
        email: String,
        password1: String,
        password2: String,
        invitation_code: Option<String>,
    ) -> FieldResult<RegisterResponse>;
    async fn token_auth(&self, email: String, password: String) -> FieldResult<TokenAuthResponse>;
    async fn refresh_token(&self, refresh_token: String) -> FieldResult<RefreshTokenResponse>;
    async fn verify_token(&self, access_token: String) -> FieldResult<VerifyTokenResponse>;
    async fn is_admin_initialized(&self) -> FieldResult<bool>;

    async fn create_invitation(&self, email: String) -> Result<i32>;
    async fn list_invitations(&self) -> Result<Vec<Invitation>>;
    async fn delete_invitation(&self, id: i32) -> Result<i32>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_generate_jwt() {
        let claims = Claims::new(UserInfo::new("test".to_string(), false));
        let token = generate_jwt(claims).unwrap();

        assert!(!token.is_empty())
    }

    #[test]
    fn test_validate_jwt() {
        let claims = Claims::new(UserInfo::new("test".to_string(), false));
        let token = generate_jwt(claims).unwrap();
        let claims = validate_jwt(&token).unwrap();
        assert_eq!(
            claims.user_info(),
            &UserInfo::new("test".to_string(), false)
        );
    }
}
