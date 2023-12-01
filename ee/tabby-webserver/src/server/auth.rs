use std::env;

use argon2::{
    password_hash,
    password_hash::{rand_core::OsRng, SaltString},
    Argon2, PasswordHasher, PasswordVerifier,
};
use async_trait::async_trait;
use jsonwebtoken as jwt;
use juniper::{FieldResult, IntoFieldError};
use lazy_static::lazy_static;
use validator::Validate;

use crate::{
    db::DbConn,
    schema::auth::{
        Claims, RefreshTokenResponse, RegisterResponse, TokenAuthResponse, UserInfo,
        ValidationErrors, VerifyTokenResponse,
    },
};

lazy_static! {
    static ref JWT_ENCODING_KEY: jwt::EncodingKey = jwt::EncodingKey::from_secret(
        jwt_token_secret().as_bytes()
    );
    static ref JWT_DECODING_KEY: jwt::DecodingKey = jwt::DecodingKey::from_secret(
        jwt_token_secret().as_bytes()
    );
    pub static ref JWT_DEFAULT_EXP: u64 = 30 * 60; // 30 minutes
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
    #[validate(must_match(
        code = "password1",
        message = "Passwords do not match",
        other = "password2"
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
    pub password2: String,
}

impl std::fmt::Debug for RegisterInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisterInput")
            .field("email", &self.email)
            .field("password1", &"********")
            .field("password2", &"********")
            .finish()
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

impl std::fmt::Debug for TokenAuthInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenAuthInput")
            .field("email", &self.email)
            .field("password", &"********")
            .finish()
    }
}

#[async_trait]
pub trait AuthenticationService {
    async fn register(&self, input: RegisterInput) -> FieldResult<RegisterResponse>;
    async fn token_auth(&self, input: TokenAuthInput) -> FieldResult<TokenAuthResponse>;
    async fn refresh_token(&self, refresh_token: String) -> FieldResult<RefreshTokenResponse>;
    async fn verify_token(&self, access_token: String) -> FieldResult<VerifyTokenResponse>;
    async fn is_admin_initialized(&self) -> FieldResult<bool>;
}

#[async_trait]
impl AuthenticationService for DbConn {
    async fn register(&self, input: RegisterInput) -> FieldResult<RegisterResponse> {
        input.validate().map_err(|err| {
            let errors = err
                .field_errors()
                .into_iter()
                .flat_map(|(_, errs)| errs)
                .cloned()
                .collect();

            ValidationErrors { errors }.into_field_error()
        })?;

        // check if email exists
        if let Some(_) = self.get_user_by_email(&input.email).await? {
            return Err("Email already exists".into());
        }

        let pwd_hash = password_hash(&input.password1)?;

        self.create_user(input.email.clone(), pwd_hash, false)
            .await?;
        let user = self.get_user_by_email(&input.email).await?.unwrap();

        let access_token = generate_jwt(Claims::new(UserInfo::new(
            user.email.clone(),
            user.is_admin,
        )))?;

        let resp = RegisterResponse::new(access_token, "".to_string());
        Ok(resp)
    }

    async fn token_auth(&self, input: TokenAuthInput) -> FieldResult<TokenAuthResponse> {
        input.validate().map_err(|err| {
            let errors = err
                .field_errors()
                .into_iter()
                .flat_map(|(_, errs)| errs)
                .cloned()
                .collect();

            ValidationErrors { errors }.into_field_error()
        })?;

        let user = self.get_user_by_email(&input.email).await?;

        let user = match user {
            Some(user) => user,
            None => return Err("User not found".into()),
        };

        if !password_verify(&input.password, &user.password_encrypted) {
            return Err("Password incorrect".into());
        }

        let access_token = generate_jwt(Claims::new(UserInfo::new(
            user.email.clone(),
            user.is_admin,
        )))?;

        let resp = TokenAuthResponse::new(access_token, "".to_string());
        Ok(resp)
    }

    async fn refresh_token(&self, _refresh_token: String) -> FieldResult<RefreshTokenResponse> {
        Ok(RefreshTokenResponse::default())
    }

    async fn verify_token(&self, access_token: String) -> FieldResult<VerifyTokenResponse> {
        let claims = validate_jwt(&access_token)?;
        let resp = VerifyTokenResponse::new(claims);
        Ok(resp)
    }

    async fn is_admin_initialized(&self) -> FieldResult<bool> {
        let admin = self.get_admin_users().await?;
        Ok(admin.len() > 0)
    }
}

fn password_hash(raw: &str) -> password_hash::Result<String> {
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let hash = argon2.hash_password(raw.as_bytes(), &salt)?.to_string();

    Ok(hash)
}

fn password_verify(raw: &str, hash: &str) -> bool {
    if let Ok(parsed_hash) = argon2::PasswordHash::new(hash) {
        let argon2 = Argon2::default();
        argon2.verify_password(raw.as_bytes(), &parsed_hash).is_ok()
    } else {
        false
    }
}

fn generate_jwt(claims: Claims) -> jwt::errors::Result<String> {
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
    env::var("TABBY_WEBSERVER_JWT_TOKEN_SECRET").unwrap_or("default_secret".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_password_hash() {
        let raw = "12345678";
        let hash = password_hash(raw).unwrap();

        assert_eq!(hash.len(), 97);
        assert!(hash.starts_with("$argon2id$v=19$m=19456,t=2,p=1$"));
    }

    #[test]
    fn test_password_verify() {
        let raw = "12345678";
        let hash = password_hash(raw).unwrap();

        assert!(password_verify(raw, &hash));
        assert!(!password_verify(raw, "invalid hash"));
    }

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
        assert_eq!(claims.user_info(), UserInfo::new("test".to_string(), false));
    }
}
