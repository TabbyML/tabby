use std::borrow::Cow;

use anyhow::{anyhow, Result};
use argon2::{
    password_hash,
    password_hash::{rand_core::OsRng, SaltString},
    Argon2, PasswordHasher, PasswordVerifier,
};
use async_trait::async_trait;
use validator::{Validate, ValidationError};

use super::db::DbConn;
use crate::schema::{
    auth::{
        generate_jwt, generate_refresh_token, validate_jwt, AuthenticationService, Invitation,
        JWTPayload, RefreshTokenError, RefreshTokenResponse, RegisterError, RegisterResponse,
        TokenAuthError, TokenAuthResponse, VerifyTokenResponse,
    },
    User,
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
    #[validate(custom = "validate_password")]
    password1: String,
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

fn validate_password(value: &str) -> Result<(), ValidationError> {
    let make_validation_error = |message: &'static str| {
        let mut err = ValidationError::new("password1");
        err.message = Some(Cow::Borrowed(message));
        Err(err)
    };

    let contains_lowercase = value.chars().any(|x| x.is_ascii_lowercase());
    if !contains_lowercase {
        return make_validation_error("Password should contains at least one lowercase character");
    }

    let contains_uppercase = value.chars().any(|x| x.is_ascii_uppercase());
    if !contains_uppercase {
        return make_validation_error("Password should contains at least one uppercase character");
    }

    let contains_digit = value.chars().any(|x| x.is_ascii_digit());
    if !contains_digit {
        return make_validation_error("Password should contains at least one numeric character");
    }

    let contains_special_char = value.chars().any(|x| x.is_ascii_punctuation());
    if !contains_special_char {
        return make_validation_error(
            "Password should contains at least one special character, e.g @#$%^&{}",
        );
    }

    Ok(())
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
    ) -> std::result::Result<RegisterResponse, RegisterError> {
        let input = RegisterInput {
            email,
            password1,
            password2,
        };
        input.validate()?;

        let is_admin_initialized = self.is_admin_initialized().await?;
        let invitation = if is_admin_initialized {
            let err = Err(RegisterError::InvalidInvitationCode);
            let Some(invitation_code) = invitation_code else {
                return err;
            };

            let Some(invitation) = self.get_invitation_by_code(&invitation_code).await? else {
                return err;
            };

            if invitation.email != input.email {
                return err;
            }

            Some(invitation)
        } else {
            None
        };

        // check if email exists
        if self.get_user_by_email(&input.email).await?.is_some() {
            return Err(RegisterError::DuplicateEmail);
        }

        let Ok(pwd_hash) = password_hash(&input.password1) else {
            return Err(RegisterError::Unknown);
        };

        let id = if let Some(invitation) = invitation {
            self.create_user_with_invitation(
                input.email.clone(),
                pwd_hash,
                !is_admin_initialized,
                invitation.id,
            )
            .await?
        } else {
            self.create_user(input.email.clone(), pwd_hash, !is_admin_initialized)
                .await?
        };

        let user = self.get_user(id).await?.unwrap();

        let refresh_token = generate_refresh_token();
        self.create_refresh_token(id, &refresh_token).await?;

        let Ok(access_token) = generate_jwt(JWTPayload::new(user.email.clone(), user.is_admin))
        else {
            return Err(RegisterError::Unknown);
        };

        let resp = RegisterResponse::new(access_token, refresh_token);
        Ok(resp)
    }

    async fn token_auth(
        &self,
        email: String,
        password: String,
    ) -> std::result::Result<TokenAuthResponse, TokenAuthError> {
        let input = TokenAuthInput { email, password };
        input.validate()?;

        let Some(user) = self.get_user_by_email(&input.email).await? else {
            return Err(TokenAuthError::UserNotFound);
        };

        if !password_verify(&input.password, &user.password_encrypted) {
            return Err(TokenAuthError::InvalidPassword);
        }

        let refresh_token = generate_refresh_token();
        self.create_refresh_token(user.id, &refresh_token).await?;

        let Ok(access_token) = generate_jwt(JWTPayload::new(user.email.clone(), user.is_admin))
        else {
            return Err(TokenAuthError::Unknown);
        };

        let resp = TokenAuthResponse::new(access_token, refresh_token);
        Ok(resp)
    }

    async fn refresh_token(
        &self,
        token: String,
    ) -> std::result::Result<RefreshTokenResponse, RefreshTokenError> {
        let Some(refresh_token) = self.get_refresh_token(&token).await? else {
            return Err(RefreshTokenError::InvalidRefreshToken);
        };
        if refresh_token.is_expired() {
            return Err(RefreshTokenError::ExpiredRefreshToken);
        }
        let Some(user) = self.get_user(refresh_token.user_id).await? else {
            return Err(RefreshTokenError::UserNotFound);
        };

        let new_token = generate_refresh_token();
        self.replace_refresh_token(&token, &new_token).await?;

        // refresh token update is done, generate new access token based on user info
        let Ok(access_token) = generate_jwt(JWTPayload::new(user.email.clone(), user.is_admin))
        else {
            return Err(RefreshTokenError::Unknown);
        };

        let resp = RefreshTokenResponse::new(access_token, new_token, refresh_token.expires_at);

        Ok(resp)
    }

    async fn verify_access_token(&self, access_token: &str) -> Result<VerifyTokenResponse> {
        let claims = validate_jwt(access_token)?;
        let resp = VerifyTokenResponse::new(claims);
        Ok(resp)
    }

    async fn is_admin_initialized(&self) -> Result<bool> {
        let admin = self.list_admin_users().await?;
        Ok(!admin.is_empty())
    }

    async fn get_user_by_email(&self, email: &str) -> Result<User> {
        let user = self.get_user_by_email(email).await?;
        if let Some(user) = user {
            Ok(user.into())
        } else {
            Err(anyhow!("User not found {}", email))
        }
    }

    async fn create_invitation(&self, email: String) -> Result<i32> {
        self.create_invitation(email).await
    }

    async fn list_invitations(&self) -> Result<Vec<Invitation>> {
        self.list_invitations().await
    }

    async fn delete_invitation(&self, id: i32) -> Result<i32> {
        self.delete_invitation(id).await
    }

    async fn reset_user_auth_token(&self, email: &str) -> Result<()> {
        self.reset_user_auth_token_by_email(email).await
    }

    async fn list_users(&self) -> Result<Vec<User>> {
        let users = self.list_users().await?;
        Ok(users.into_iter().map(|x| x.into()).collect())
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
    use assert_matches::assert_matches;

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

    static ADMIN_EMAIL: &str = "test@example.com";
    static ADMIN_PASSWORD: &str = "123456789$acR";

    async fn register_admin_user(conn: &DbConn) -> RegisterResponse {
        conn.register(
            ADMIN_EMAIL.to_owned(),
            ADMIN_PASSWORD.to_owned(),
            ADMIN_PASSWORD.to_owned(),
            None,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_auth_token() {
        let conn = DbConn::new_in_memory().await.unwrap();
        assert_matches!(
            conn.token_auth(ADMIN_EMAIL.to_owned(), "12345678".to_owned())
                .await,
            Err(TokenAuthError::UserNotFound)
        );

        register_admin_user(&conn).await;

        assert_matches!(
            conn.token_auth(ADMIN_EMAIL.to_owned(), "12345678".to_owned())
                .await,
            Err(TokenAuthError::InvalidPassword)
        );

        let resp1 = conn
            .token_auth(ADMIN_EMAIL.to_owned(), ADMIN_PASSWORD.to_owned())
            .await
            .unwrap();
        let resp2 = conn
            .token_auth(ADMIN_EMAIL.to_owned(), ADMIN_PASSWORD.to_owned())
            .await
            .unwrap();
        // each auth should generate a new refresh token
        assert_ne!(resp1.refresh_token, resp2.refresh_token);
    }

    #[tokio::test]
    async fn test_invitation_flow() {
        let conn = DbConn::new_in_memory().await.unwrap();

        assert!(!conn.is_admin_initialized().await.unwrap());
        register_admin_user(&conn).await;

        let email = "user@user.com";
        let password = "12345678dD^";

        conn.create_invitation(email.to_owned()).await.unwrap();
        let invitation = &conn.list_invitations().await.unwrap()[0];

        // Admin initialized, registeration requires a invitation code;
        assert_matches!(
            conn.register(
                email.to_owned(),
                password.to_owned(),
                password.to_owned(),
                None
            )
            .await,
            Err(RegisterError::InvalidInvitationCode)
        );

        // Invalid invitation code won't work.
        assert_matches!(
            conn.register(
                email.to_owned(),
                password.to_owned(),
                password.to_owned(),
                Some("abc".to_owned())
            )
            .await,
            Err(RegisterError::InvalidInvitationCode)
        );

        // Register success.
        assert!(conn
            .register(
                email.to_owned(),
                password.to_owned(),
                password.to_owned(),
                Some(invitation.code.clone())
            )
            .await
            .is_ok());

        // Try register again with same email failed.
        assert_matches!(
            conn.register(
                email.to_owned(),
                password.to_owned(),
                password.to_owned(),
                Some(invitation.code.clone())
            )
            .await,
            Err(RegisterError::InvalidInvitationCode)
        );

        // Used invitation should have been deleted,  following delete attempt should fail.
        assert!(conn.delete_invitation(invitation.id).await.is_err());
    }

    #[tokio::test]
    async fn test_refresh_token() {
        let conn = DbConn::new_in_memory().await.unwrap();
        let reg = register_admin_user(&conn).await;

        let resp1 = conn.refresh_token(reg.refresh_token.clone()).await.unwrap();
        // new access token should be valid
        assert!(validate_jwt(&resp1.access_token).is_ok());
        // refresh token should be renewed
        assert_ne!(reg.refresh_token, resp1.refresh_token);

        let resp2 = conn
            .refresh_token(resp1.refresh_token.clone())
            .await
            .unwrap();
        // expire time should be no change
        assert_eq!(resp1.refresh_expires_at, resp2.refresh_expires_at);
    }

    #[tokio::test]
    async fn test_reset_user_auth_token() {
        let conn = DbConn::new_in_memory().await.unwrap();
        register_admin_user(&conn).await;

        let user = conn.get_user_by_email(ADMIN_EMAIL).await.unwrap().unwrap();
        conn.reset_user_auth_token(&user.email).await.unwrap();

        let user2 = conn.get_user_by_email(ADMIN_EMAIL).await.unwrap().unwrap();
        assert_ne!(user.auth_token, user2.auth_token);
    }
}
