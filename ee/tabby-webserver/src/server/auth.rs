use std::env;

use anyhow::Result;
use argon2::{
    password_hash,
    password_hash::{rand_core::OsRng, SaltString},
    Argon2, PasswordHasher, PasswordVerifier,
};
use async_trait::async_trait;
use jsonwebtoken as jwt;
use lazy_static::lazy_static;
use validator::Validate;

use crate::{
    db::DbConn,
    schema::auth::{
        AuthError, Claims, RefreshTokenResponse, RegisterResponse, TokenAuthResponse, UserInfo,
        VerifyTokenResponse,
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

#[derive(Validate)]
pub struct RegisterInput {
    #[validate(email)]
    #[validate(length(
        max = 128,
        code = "email_too_long",
        message = "Email must be at most 128 characters"
    ))]
    pub email: String,
    #[validate(length(
        min = 8,
        code = "password_too_short",
        message = "Password must be at least 8 characters"
    ))]
    #[validate(length(
        max = 20,
        code = "password_too_long",
        message = "Password must be at most 20 characters"
    ))]
    #[validate(must_match(
        other = "password2",
        code = "password_mismatch",
        message = "Passwords do not match"
    ))]
    pub password1: String,
    #[validate(length(min = 8, max = 20))]
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

#[derive(Validate)]
pub struct TokenAuthInput {
    #[validate(email)]
    #[validate(length(
        max = 128,
        code = "email_too_long",
        message = "Email must be at most 128 characters"
    ))]
    pub email: String,
    #[validate(length(
        min = 8,
        code = "password_too_short",
        message = "Password must be at least 8 characters"
    ))]
    #[validate(length(
        max = 20,
        code = "password_too_long",
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
    async fn register(&self, input: RegisterInput) -> Result<RegisterResponse>;
    async fn token_auth(&self, input: TokenAuthInput) -> Result<TokenAuthResponse>;
    async fn refresh_token(&self, refresh_token: String) -> Result<RefreshTokenResponse>;
    async fn verify_token(&self, access_token: String) -> Result<VerifyTokenResponse>;
}

#[async_trait]
impl AuthenticationService for DbConn {
    async fn register(&self, input: RegisterInput) -> Result<RegisterResponse> {
        if let Err(err) = input.validate() {
            let mut errors = vec![];
            for (_, errs) in err.field_errors() {
                errors.extend(errs.iter().map(|e| e.clone().into()));
            }
            let resp = RegisterResponse::with_errors(errors);
            return Ok(resp);
        }

        // check if email exists
        if let Some(_) = self.get_user_by_email(&input.email).await? {
            let resp = RegisterResponse::with_error(AuthError {
                message: "Email already exists".to_string(),
                code: "email_already_exists".to_string(),
            });
            return Ok(resp);
        }

        let pwd_hash = match password_hash(&input.password1) {
            Ok(hash) => hash,
            Err(err) => {
                return Ok(RegisterResponse::with_error(err.into()));
            }
        };

        self.create_user(input.email.clone(), pwd_hash, false)
            .await?;
        let user = self.get_user_by_email(&input.email).await?.unwrap();

        let access_token = match generate_jwt(Claims::new(UserInfo::new(
            user.email.clone(),
            user.is_admin,
        ))) {
            Ok(token) => token,
            Err(err) => {
                return Ok(RegisterResponse::with_error(err.into()));
            }
        };

        let resp = RegisterResponse::new(access_token, "".to_string());
        Ok(resp)
    }

    async fn token_auth(&self, input: TokenAuthInput) -> Result<TokenAuthResponse> {
        if let Err(err) = input.validate() {
            let mut errors = vec![];
            for (_, errs) in err.field_errors() {
                errors.extend(errs.iter().map(|e| e.clone().into()));
            }
            let resp = TokenAuthResponse::with_errors(errors);
            return Ok(resp);
        }

        let user = self.get_user_by_email(&input.email).await?;

        let user = match user {
            Some(user) => user,
            None => {
                let resp = TokenAuthResponse::with_error(AuthError {
                    message: "User not found".to_string(),
                    code: "user_not_found".to_string(),
                });
                return Ok(resp);
            }
        };

        if !password_verify(&input.password, &user.password_encrypted) {
            let resp = TokenAuthResponse::with_error(AuthError {
                message: "Incorrect password".to_string(),
                code: "incorrect_password".to_string(),
            });
            return Ok(resp);
        }

        let access_token = match generate_jwt(Claims::new(UserInfo::new(
            user.email.clone(),
            user.is_admin,
        ))) {
            Ok(token) => token,
            Err(err) => {
                return Ok(TokenAuthResponse::with_error(err.into()));
            }
        };

        let resp = TokenAuthResponse::new(access_token, "".to_string());
        Ok(resp)
    }

    async fn refresh_token(&self, _refresh_token: String) -> Result<RefreshTokenResponse> {
        Ok(RefreshTokenResponse::default())
    }

    async fn verify_token(&self, access_token: String) -> Result<VerifyTokenResponse> {
        let claims = match validate_jwt(&access_token) {
            Ok(claims) => claims,
            Err(err) => {
                return Ok(VerifyTokenResponse::with_error(err.into()));
            }
        };

        let resp = VerifyTokenResponse::new(claims);
        Ok(resp)
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
