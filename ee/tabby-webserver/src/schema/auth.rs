use std::fmt::Debug;

use jsonwebtoken as jwt;
use juniper::{FieldError, GraphQLObject, IntoFieldError, Object, ScalarValue, Value};
use serde::{Deserialize, Serialize};
use validator::ValidationError;

use crate::server::auth::JWT_DEFAULT_EXP;

#[derive(Debug)]
pub struct ValidationErrors {
    pub errors: Vec<ValidationError>,
}

impl<S: ScalarValue> IntoFieldError<S> for ValidationErrors {
    fn into_field_error(self) -> FieldError<S> {
        let errors = self
            .errors
            .into_iter()
            .map(|err| {
                let mut obj = Object::with_capacity(2);
                obj.add_field("path", Value::scalar(err.code.to_string()));
                obj.add_field(
                    "message",
                    Value::scalar(err.message.unwrap_or_default().to_string()),
                );
                obj.into()
            })
            .collect::<Vec<_>>();
        let mut ext = Object::with_capacity(2);
        ext.add_field("code", Value::scalar("validation-error".to_string()));
        ext.add_field("errors", Value::list(errors));

        FieldError::new("Invalid input parameters", ext.into())
    }
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

    pub fn user_info<'a>(&'a self) -> &'a UserInfo {
        &self.user
    }
}
