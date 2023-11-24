use std::borrow::Cow;

use argon2::password_hash;
use jsonwebtoken as jwt;
use juniper::GraphQLObject;
use serde::{Deserialize, Serialize};
use validator::ValidationError;

use crate::server::auth::JWT_DEFAULT_EXP;

#[derive(Debug, GraphQLObject)]
pub struct AuthError {
    pub message: String,
    pub code: String,
}

impl From<ValidationError> for AuthError {
    fn from(err: ValidationError) -> Self {
        Self {
            message: err.message.unwrap_or(Cow::from("unknown error")).into(),
            code: err.code.to_string(),
        }
    }
}

impl From<password_hash::Error> for AuthError {
    fn from(err: password_hash::Error) -> Self {
        Self {
            message: err.to_string(),
            code: "password_hash_error".to_string(),
        }
    }
}

impl From<jwt::errors::Error> for AuthError {
    fn from(err: jwt::errors::Error) -> Self {
        Self {
            message: err.to_string(),
            code: "jwt_error".to_string(),
        }
    }
}

#[derive(Debug, GraphQLObject)]
pub struct RegisterResponse {
    access_token: String,
    refresh_token: String,
    errors: Vec<AuthError>,
}

impl RegisterResponse {
    pub fn new(access_token: String, refresh_token: String) -> Self {
        Self {
            access_token,
            refresh_token,
            errors: vec![],
        }
    }

    pub fn with_error(error: AuthError) -> Self {
        Self {
            access_token: "".to_string(),
            refresh_token: "".to_string(),
            errors: vec![error],
        }
    }

    pub fn with_errors(errors: Vec<AuthError>) -> Self {
        Self {
            access_token: "".to_string(),
            refresh_token: "".to_string(),
            errors,
        }
    }
}

#[derive(Debug, GraphQLObject)]
pub struct TokenAuthResponse {
    access_token: String,
    refresh_token: String,
    errors: Vec<AuthError>,
}

impl TokenAuthResponse {
    pub fn new(access_token: String, refresh_token: String) -> Self {
        Self {
            access_token,
            refresh_token,
            errors: vec![],
        }
    }

    pub fn with_error(error: AuthError) -> Self {
        Self {
            access_token: "".to_string(),
            refresh_token: "".to_string(),
            errors: vec![error],
        }
    }

    pub fn with_errors(errors: Vec<AuthError>) -> Self {
        Self {
            access_token: "".to_string(),
            refresh_token: "".to_string(),
            errors,
        }
    }
}

#[derive(Debug, Default, GraphQLObject)]
pub struct RefreshTokenResponse {
    access_token: String,
    refresh_token: String,
    refresh_expires_in: i32,
    errors: Vec<AuthError>,
}

#[derive(Debug, GraphQLObject)]
pub struct VerifyTokenResponse {
    errors: Vec<AuthError>,
    claims: Claims,
}

impl VerifyTokenResponse {
    pub fn new(claims: Claims) -> Self {
        Self {
            errors: vec![],
            claims,
        }
    }

    pub fn with_error(error: AuthError) -> Self {
        Self {
            errors: vec![error],
            claims: Claims::default(),
        }
    }

    pub fn with_errors(errors: Vec<AuthError>) -> Self {
        Self {
            errors,
            claims: Claims::default(),
        }
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

    pub fn user_info(self) -> UserInfo {
        self.user
    }
}
