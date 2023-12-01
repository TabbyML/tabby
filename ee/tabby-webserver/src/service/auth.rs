use argon2::{
    password_hash,
    password_hash::{rand_core::OsRng, SaltString},
    Argon2, PasswordHasher, PasswordVerifier,
};
use async_trait::async_trait;
use juniper::{FieldResult, IntoFieldError};
use validator::Validate;

use super::db::DbConn;
use crate::schema::{
    auth::{
        generate_jwt, validate_jwt, AuthenticationService, Claims, RefreshTokenResponse,
        RegisterResponse, TokenAuthResponse, UserInfo, VerifyTokenResponse,
    },
    ValidationErrors,
};

/// Input parameters for register mutation
/// `validate` attribute is used to validate the input parameters
///   - `code` argument specifies which parameter causes the failure
///   - `message` argument provides client friendly error message
///
#[derive(Validate)]
struct RegisterInput {
    #[validate(email(code = "email", message = "Email is invalid"))]
    #[validate(length(
        max = 128,
        code = "email",
        message = "Email must be at most 128 characters"
    ))]
    email: String,
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
    password1: String,
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
    password2: String,
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
struct TokenAuthInput {
    #[validate(email(code = "email", message = "Email is invalid"))]
    #[validate(length(
        max = 128,
        code = "email",
        message = "Email must be at most 128 characters"
    ))]
    email: String,
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
    password: String,
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
impl AuthenticationService for DbConn {
    async fn register(
        &self,
        email: String,
        password1: String,
        password2: String,
        invitation_code: Option<String>,
    ) -> FieldResult<RegisterResponse> {
        let input = RegisterInput {
            email,
            password1,
            password2,
        };
        input.validate().map_err(|err| {
            let errors = err
                .field_errors()
                .into_iter()
                .flat_map(|(_, errs)| errs)
                .cloned()
                .collect();

            ValidationErrors { errors }.into_field_error()
        })?;

        if self.is_admin_initialized().await? {
            let err = Err("Invitation code is not valid".into());
            let Some(invitation_code) = invitation_code else {
                return err;
            };

            let Some(invitation) = self.get_invitation_by_code(invitation_code).await? else {
                return err;
            };

            if invitation.email != input.email {
                return err;
            }
        };

        // check if email exists
        if self.get_user_by_email(&input.email).await?.is_some() {
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

    async fn token_auth(&self, email: String, password: String) -> FieldResult<TokenAuthResponse> {
        let input = TokenAuthInput { email, password };
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
        let admin = self.list_admin_users().await?;
        Ok(!admin.is_empty())
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
}
